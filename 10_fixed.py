"""
Video Diffusion Model Fine-tuning Script
========================================

This script implements a complete video diffusion model fine-tuning pipeline
based on the Tune-A-Video approach. It loads pre-trained Stable Diffusion models,
converts them to 3D UNet for video generation, and fine-tunes specific attention modules.

Key Features:
- Loads pre-trained Stable Diffusion models from Hugging Face Hub
- Converts 2D UNet to 3D UNet for video processing
- Fine-tunes specific attention modules (attn1.to_q, attn2.to_q, attn_temp)
- Uses Accelerate for distributed training and mixed precision
- Implements validation and checkpointing during training
- Supports gradient accumulation and memory optimizations

Author: Deep Learning Researcher
Date: 2024
"""

import os
import math
import json
from typing import Dict, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader

# Import custom models
from models.unet import UNet3DConditionModel
from models.pipeline_tuneavideo import TuneAVideoPipeline
from utils.util import save_videos_grid

# Set PyTorch CUDA memory allocation strategy to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data paths
    "frames_path": "data/metadata/videos_frames.npy",
    "prompt_ids_path": "data/metadata/video_descriptions_en_short_prompt_ids.pt",
    
    # Training parameters
    "seed": 42,
    "BATCH_SIZE": 32,
    "use_8bit_adam": True,
    "learning_rate": 3e-5,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "no",  # "no", "fp16", "bf16"
    
    # Model configuration
    "pretrained_model_path": "CompVis/stable-diffusion-v1-4",
    "trainable_modules": [
        "attn1.to_q",
        "attn2.to_q",
        "attn_temp",
    ],
    "enable_xformers_memory_efficient_attention": True,
    "gradient_checkpointing": True,
    
    # Training schedule
    "num_train_epochs": 300,
    "validation_steps": 100,
    "checkpointing_steps": 100,
    "max_grad_norm": 1.0,
    
    # Save paths
    "save_path": "checkpoints/video_diffusion_finetuned",
    "output_dir": "checkpoints/video_diffusion_finetuned",
    
    # Validation data
    "validation_data": {
        "prompts": [
            "a teddy bear is playing the guitar",
            "a panda is surfing on the sea"
        ],
        "video_length": 8,
        "width": 512,
        "height": 512,
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
    },
    
    # Optimizer parameters
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_weight_decay": 1e-2,
    "adam_epsilon": 1e-08,
    
    # LR scheduler parameters
    "lr_scheduler": "constant",
    "lr_warmup_steps": 0,
}

print("✅ Configuration loaded successfully.")

# ============================================================================
# DATASET DEFINITION
# ============================================================================

class TuneMultiVideoDataset(Dataset):
    """
    Dataset for loading pre-processed video frames and prompt IDs.
    
    This dataset loads pre-extracted video frames and their corresponding
    tokenized text prompts for efficient training.
    """
    
    def __init__(self, frames_path: str, prompt_ids_path: str):
        """
        Initialize the dataset.
        
        Args:
            frames_path (str): Path to the pre-processed video frames numpy file
            prompt_ids_path (str): Path to the tokenized prompt IDs torch file
        """
        print(f"Loading frames from: {frames_path}")
        print(f"Loading prompt IDs from: {prompt_ids_path}")
        
        self.frames = np.load(frames_path)  # Expected shape: (250, 12, 3, 288, 512)
        self.prompt_ids = torch.load(prompt_ids_path)  # Expected shape: (250, 77)
        
        print(f"Loaded {len(self.frames)} video samples")
        print(f"Frames shape: {self.frames.shape}")
        print(f"Prompt IDs shape: {self.prompt_ids.shape}")
        
        # Validate data consistency
        assert len(self.frames) == len(self.prompt_ids), \
            f"Number of frames ({len(self.frames)}) must match number of prompts ({len(self.prompt_ids)})"

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        """
        Get a single training sample.
        
        Args:
            index (int): Sample index
            
        Returns:
            dict: Dictionary containing normalized frames and prompt IDs
        """
        # Normalize pixel values to [-1, 1] range
        norm_frames = (self.frames[index] / 127.5) - 1.0
        
        return {
            "pixel_values": torch.from_numpy(norm_frames).float(),
            "prompt_ids": self.prompt_ids[index]
        }

def prepare_dataloader(frames_path: str, prompt_ids_path: str, batch_size: int) -> DataLoader:
    """
    Prepare the data loader for training.
    
    Args:
        frames_path (str): Path to video frames
        prompt_ids_path (str): Path to prompt IDs
        batch_size (int): Batch size for training
        
    Returns:
        DataLoader: Configured data loader
    """
    train_dataset = TuneMultiVideoDataset(frames_path, prompt_ids_path)
    dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    return dataloader

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models(model_repo_id: str, trainable_modules: Tuple[str, ...]) -> Tuple:
    """
    Load all necessary models from Hugging Face Hub and set trainable parameters.
    
    Args:
        model_repo_id (str): Hugging Face Hub model repository ID
        trainable_modules (Tuple[str, ...]): UNet module name suffixes to unfreeze for training
        
    Returns:
        Tuple: (tokenizer, text_encoder, vae, unet, noise_scheduler)
    """
    print(f"Loading models from Hugging Face Hub: '{model_repo_id}'")
    
    try:
        # Load all model components
        noise_scheduler = DDPMScheduler.from_pretrained(model_repo_id, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(model_repo_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_repo_id, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(model_repo_id, subfolder="vae")
        
        # Load and convert 2D UNet to 3D UNet
        unet = UNet3DConditionModel.from_pretrained_2d(model_repo_id, subfolder="unet")
        
        print("✅ All models loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading models from '{model_repo_id}': {e}")
        print("Please check the model ID and your network connection.")
        raise e

    # Freeze models that don't need training
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Unfreeze only the specified trainable modules
    print("Unfreezing the following modules for fine-tuning:")
    for name, module in unet.named_modules():
        if any(name.endswith(trainable_module) for trainable_module in trainable_modules):
            for param in module.parameters():
                param.requires_grad = True
            print(f"  - {name}")
    
    return tokenizer, text_encoder, vae, unet, noise_scheduler

# ============================================================================
# OPTIMIZER CREATION
# ============================================================================

def create_optimizer(unet: UNet3DConditionModel) -> torch.optim.Optimizer:
    """
    Create optimizer for the trainable parameters.
    
    Args:
        unet (UNet3DConditionModel): The UNet model with trainable parameters
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    # Choose optimizer class based on configuration
    optimizer_cls = torch.optim.AdamW
    if CONFIG["use_8bit_adam"]:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            print("✅ Using 8-bit Adam optimizer")
        except ImportError:
            print("⚠️  bitsandbytes not available, falling back to regular AdamW")
            print("   Install with: pip install bitsandbytes")
    
    # Get only trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, unet.parameters())
    
    optimizer = optimizer_cls(
        trainable_params,
        lr=CONFIG["learning_rate"],
        betas=(CONFIG["adam_beta1"], CONFIG["adam_beta2"]),
        weight_decay=CONFIG["adam_weight_decay"],
        eps=CONFIG["adam_epsilon"]
    )
    
    return optimizer

# ============================================================================
# VALIDATION AND CHECKPOINTING
# ============================================================================

def validate_and_save_checkpoint(
    epoch: int, 
    accelerator: Accelerator, 
    unet: UNet3DConditionModel, 
    vae: AutoencoderKL, 
    text_encoder: CLIPTextModel, 
    tokenizer: CLIPTokenizer, 
    noise_scheduler: DDPMScheduler
):
    """
    Run validation and save checkpoint if needed.
    
    Args:
        epoch (int): Current training epoch
        accelerator (Accelerator): Accelerate instance
        unet (UNet3DConditionModel): UNet model
        vae (AutoencoderKL): VAE model
        text_encoder (CLIPTextModel): Text encoder
        tokenizer (CLIPTokenizer): Tokenizer
        noise_scheduler (DDPMScheduler): Noise scheduler
    """
    if not accelerator.is_main_process:
        return
        
    print(f"\n🔄 Running validation for epoch {epoch + 1}...")
    
    # Create pipeline for inference
    unet_for_pipeline = accelerator.unwrap_model(unet)
    pipeline = TuneAVideoPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet_for_pipeline,
        scheduler=DDIMScheduler.from_config(noise_scheduler.config)
    )
    
    # Enable VAE slicing for memory efficiency
    pipeline.enable_vae_slicing()
    
    # Set up generator for reproducible results
    generator = torch.Generator(device=accelerator.device).manual_seed(CONFIG["seed"])
    
    # Create samples directory
    samples_dir = os.path.join(CONFIG["save_path"], "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Generate validation videos
    for i, prompt in enumerate(CONFIG["validation_data"]["prompts"]):
        print(f"  Generating video for prompt {i+1}: '{prompt}'")
        
        with torch.autocast("cuda"):
            video = pipeline(
                prompt, 
                generator=generator, 
                **CONFIG["validation_data"]
            ).videos
        
        # Save video
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in " _-")[:30]
        save_path = os.path.join(
            samples_dir, 
            f"epoch-{epoch+1:04d}_prompt-{i:02d}_{safe_prompt}.gif"
        )
        save_videos_grid(video, save_path)
        print(f"    Saved to: {save_path}")
    
    print(f"✅ Validation samples saved for epoch {epoch+1}")
    
    # Save checkpoint if needed
    if (epoch + 1) % CONFIG["checkpointing_steps"] == 0:
        checkpoint_dir = os.path.join(CONFIG["save_path"], f"checkpoint-{epoch+1}")
        pipeline.save_pretrained(checkpoint_dir)
        print(f"💾 Checkpoint saved to: {checkpoint_dir}")

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_one_epoch(
    unet: UNet3DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    noise_scheduler: DDPMScheduler,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dataloader: DataLoader,
    accelerator: Accelerator,
    weight_dtype: torch.dtype
) -> float:
    """
    Train for one epoch.
    
    Args:
        unet (UNet3DConditionModel): UNet model
        vae (AutoencoderKL): VAE model
        text_encoder (CLIPTextModel): Text encoder
        noise_scheduler (DDPMScheduler): Noise scheduler
        optimizer (torch.optim.Optimizer): Optimizer
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        train_dataloader (DataLoader): Training data loader
        accelerator (Accelerator): Accelerate instance
        weight_dtype (torch.dtype): Data type for computations
        
    Returns:
        float: Average loss for the epoch
    """
    unet.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_dataloader:
        with accelerator.accumulate(unet):
            # Prepare input data
            pixel_values = batch["pixel_values"].to(weight_dtype)
            prompt_ids = batch["prompt_ids"]
            
            video_length = pixel_values.shape[1]
            
            # Encode video frames to latent space
            with torch.no_grad():
                # Flatten frames for VAE encoding: (b, f, c, h, w) -> (b*f, c, h, w)
                pixel_values_flat = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values_flat).latent_dist.sample()
                # Reshape back: (b*f, c, h, w) -> (b, c, f, h, w)
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * vae.config.scaling_factor
                
                # Encode text prompts
                encoder_hidden_states = text_encoder(prompt_ids)[0]
            
            # Add noise to latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, 
                noise_scheduler.config.num_train_timesteps, 
                (bsz,), 
                device=latents.device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Determine target based on prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            else:  # v_prediction
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            
            # Forward pass
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # Calculate loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
            # Backward pass
            accelerator.backward(loss)
            
            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, unet.parameters()), 
                    CONFIG["max_grad_norm"]
                )
            
            # Optimizer step
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.detach().item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """
    Main training function that orchestrates the entire training process.
    """
    print("🚀 Starting Video Diffusion Model Fine-tuning")
    print("=" * 60)
    
    # 1. Initialize Accelerator
    print("📦 Initializing Accelerator...")
    accelerator = Accelerator(
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        mixed_precision=CONFIG["mixed_precision"],
    )
    
    # 2. Set random seed
    if CONFIG["seed"] is not None:
        set_seed(CONFIG["seed"])
        print(f"🎲 Random seed set to: {CONFIG['seed']}")
    
    # 3. Load models
    print("🤖 Loading pre-trained models...")
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_models(
        CONFIG["pretrained_model_path"], 
        tuple(CONFIG["trainable_modules"])
    )
    
    # 4. Apply optimizations
    print("⚡ Applying optimizations...")
    if CONFIG["enable_xformers_memory_efficient_attention"]:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            print("  ✅ xformers memory efficient attention enabled")
        else:
            print("  ⚠️  xformers not available, skipping")
    
    if CONFIG["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
        print("  ✅ Gradient checkpointing enabled")
    
    # 5. Prepare data, optimizer, and scheduler
    print("📊 Preparing data and optimizers...")
    train_dataloader = prepare_dataloader(
        CONFIG["frames_path"],
        CONFIG["prompt_ids_path"],
        CONFIG["BATCH_SIZE"]
    )
    
    optimizer = create_optimizer(unet)
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / CONFIG["gradient_accumulation_steps"])
    num_training_steps = CONFIG["num_train_epochs"] * num_update_steps_per_epoch
    
    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        CONFIG["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=CONFIG["lr_warmup_steps"] * CONFIG["gradient_accumulation_steps"],
        num_training_steps=num_training_steps * accelerator.num_processes,
    )
    
    # 6. Prepare all components with Accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # 7. Set data types and move models to device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    # 8. Create output directories
    os.makedirs(CONFIG["save_path"], exist_ok=True)
    os.makedirs(os.path.join(CONFIG["save_path"], "samples"), exist_ok=True)
    
    # 9. Print training information
    if accelerator.is_main_process:
        print("\n📋 Training Configuration:")
        print(f"   Total epochs: {CONFIG['num_train_epochs']}")
        print(f"   Samples per epoch: {len(train_dataloader.dataset)}")
        print(f"   Batch size per device: {CONFIG['BATCH_SIZE']}")
        print(f"   Gradient accumulation steps: {CONFIG['gradient_accumulation_steps']}")
        print(f"   Learning rate: {CONFIG['learning_rate']}")
        print(f"   Mixed precision: {CONFIG['mixed_precision']}")
        print(f"   Save path: {CONFIG['save_path']}")
        print()
    
    # 10. Training loop
    print("🔥 Starting training loop...")
    for epoch in range(CONFIG["num_train_epochs"]):
        # Create progress bar
        progress_bar = tqdm(
            range(len(train_dataloader)),
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}/{CONFIG['num_train_epochs']}"
        )
        
        # Train for one epoch
        avg_loss = train_one_epoch(
            unet, vae, text_encoder, noise_scheduler, optimizer,
            lr_scheduler, train_dataloader, accelerator, weight_dtype
        )
        
        # Update progress bar
        progress_bar.set_postfix(
            loss=f"{avg_loss:.4f}",
            lr=f"{lr_scheduler.get_last_lr()[0]:.2e}"
        )
        
        # Validation and checkpointing
        if (epoch + 1) % CONFIG["validation_steps"] == 0:
            validate_and_save_checkpoint(
                epoch, accelerator, unet, vae, text_encoder, 
                tokenizer, noise_scheduler
            )
    
    # 11. Final save
    print("\n💾 Saving final model...")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = TuneAVideoPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=DDIMScheduler.from_config(noise_scheduler.config)
        )
        pipeline.save_pretrained(CONFIG["output_dir"])
        print(f"🎉 Training completed! Final model saved to: {CONFIG['output_dir']}")

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()
