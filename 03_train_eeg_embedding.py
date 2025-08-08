"""
EEG Embedding Training Script

This script trains an EEG embedding model that transforms EEG signals into a low-dimensional 
semantic representation. The ground truth semantic representation is obtained by:
1. Using gemini-2.5-flash to transform videos to descriptive text
2. Using "openai/clip-vit-large-patch14" to transform text into embedding vectors

Author: Deep Learning Research Team
Date: 2025
"""

import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from einops import rearrange, repeat
from sklearn import preprocessing
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eeg_embedding_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "eeg_data_path": "data/PSD_DE/imaging",  # or "data/PSD_DE/watching"
    "text_embedding_path": "data/metadata/text_embedding.pt",
    "save_checkpoint_path": "checkpoints/eeg_embedding",
    
    "seed": 42,
    "train_valid": [0.8, 0.2], 
    "batch_size": 32, 
    "num_workers": 0,
    
    "epochs": 200, 
    "learning_rate": 5e-4,
    "valid_steps": 10  # Validate every 10 epochs
}

def set_seed(seed):
    """Sets random seeds for reproducibility across all libraries.

    Args:
        seed (int): The seed to use for all random number generators.
    """
    logger.info(f"Setting random seed to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class EEGTextDataset(Dataset):
    """PyTorch Dataset for EEG and corresponding text embeddings.

    Args:
        eeg (np.ndarray): A numpy array of EEG data, with shape
                          (n_samples, n_features).
        text (torch.Tensor): A torch tensor of text embeddings, with shape
                             (n_samples, n_embedding_dim).
    """
    def __init__(self, eeg: np.ndarray, text: torch.Tensor):
        self.eeg = eeg
        self.text = text
        self.len = eeg.shape[0]
        logger.info(f"Dataset initialized with {self.len} samples")

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, item: int) -> tuple[np.ndarray, torch.Tensor]:
        return self.eeg[item], self.text[item]

def prepare_dataloader(eeg_data_path: str, text_embed_path: str,
                       batch_size: int, num_workers: int):
    """Loads, preprocesses, and prepares the EEG and Text data.

    This function handles:
    1. Loading the raw EEG data from multiple files.
    2. Reshaping and normalizing the EEG data.
    3. Loading corresponding text embeddings.
    4. Creating train/validation DataLoaders.

    Args:
        eeg_data_path (str): Path to the directory containing EEG numpy files.
        text_embed_path (str): Path to the text embedding file.
        batch_size (int): The batch size for the DataLoader.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        tuple: (train_loader, valid_loader) DataLoaders for training and validation.
    """
    logger.info("Preparing data...")
    
    # Load EEG data from all files in the directory
    eeg_data = []
    eeg_files = [f for f in os.listdir(eeg_data_path) if f.endswith('.npy')]
    logger.info(f"Found {len(eeg_files)} EEG files to load")
    
    for file in eeg_files:
        file_path = os.path.join(eeg_data_path, file)
        data = np.load(file_path)
        eeg_data.append(data)
        logger.debug(f"Loaded {file} with shape: {data.shape}")
    
    eeg_data = np.stack(eeg_data, axis=0)
    logger.info(f"Loaded EEG data with shape: {eeg_data.shape}")  # Expected: (6, 2, 5, 50, 62, 5)

    # Process each session's EEG data - reshape to (samples, features)
    eeg = rearrange(eeg_data, 'a b c d e f -> (a b c d) (e f)')  # shape(60*2*5*50, 62*5)
    logger.info(f"Reshaped EEG data to: {eeg.shape}")
    
    # Load and process text embeddings
    text = torch.load(text_embed_path)  # Expected shape: (250, 77, 768)
    logger.info(f"Loaded text embeddings with shape: {text.shape}")
    
    # Repeat text embeddings to match EEG samples
    text = repeat(text, 'a b c -> (num a) (b c)', num=eeg_data.shape[0]*2)  # shape(60*2*250, 59136)
    logger.info(f"Reshaped text embeddings to: {text.shape}")

    # Normalize EEG data using StandardScaler
    logger.info("Normalizing EEG data...")
    normalize = preprocessing.StandardScaler()
    eeg = normalize.fit_transform(eeg)
    logger.info(f"Final EEG data shape: {eeg.shape}")
    logger.info(f"Final Text embedding shape: {text.shape}")

    # Create dataset and split into train/validation
    dataset = EEGTextDataset(eeg, text)
    trainset, validset = random_split(dataset, CONFIG["train_valid"])
    logger.info(f"Split dataset: {len(trainset)} training samples, {len(validset)} validation samples")
    
    # Create DataLoaders
    train_loader = DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()  # Only pin memory if CUDA is available
    )
    
    valid_loader = DataLoader(
        validset, 
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        drop_last=False,
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, valid_loader

class EEG_Embedding(nn.Module):
    """A multi-layer perceptron model to map EEG signals to text embedding space.

    This model takes flattened EEG data and projects it into a high-dimensional
    space matching the dimensions of CLIP's text embeddings (77 * 768 = 59136).

    Architecture:
        - Input: EEG features (310 dimensions)
        - Hidden layers: 1000 -> 10000 -> 1000 -> 1000 dimensions
        - Output: 59136 dimensions (77 * 768, matching CLIP text embeddings)
    """
    def __init__(self):
        super(EEG_Embedding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(310, 1000),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(1000, 10000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(10000, 1000),  # Fixed: was 1000 -> 1000, should be 10000 -> 1000
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 77 * 768)  # Output dimension: 59136
        )
        logger.info("EEG_Embedding model initialized")

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            eeg (torch.Tensor): A batch of EEG data with shape (batch_size, 310).

        Returns:
            torch.Tensor: The predicted text embeddings with shape
                          (batch_size, 77 * 768).
        """
        eeg_embeddings = self.mlp(eeg)
        return eeg_embeddings

def loss_fn(batch, model, criterion, device): 
    """Compute loss for a batch of EEG and text data.
    
    Args:
        batch: Tuple of (eeg_data, text_embeddings) from DataLoader
        model: The EEG embedding model
        criterion: Loss function (e.g., MSE)
        device: Device to run computation on
        
    Returns:
        torch.Tensor: Computed loss value
    """
    eeg_data, text_embeddings = batch
    eeg_data = eeg_data.float().to(device)  # Convert to float and move to device
    text_embeddings = text_embeddings.float().to(device)
    
    # Forward pass through model
    predicted_embeddings = model(eeg_data)
    
    # Compute loss
    loss = criterion(predicted_embeddings, text_embeddings)
    return loss

def validate_model(data_loader, model, criterion, device): 
    """Validate the model on the validation set.
    
    Args:
        data_loader: Validation DataLoader
        model: The model to validate
        criterion: Loss function
        device: Device to run computation on
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    running_loss = 0
    num_batches = len(data_loader)
    
    logger.info("Starting validation...")
    pbar = tqdm(total=len(data_loader.dataset), ncols=0, desc='Valid')

    with torch.no_grad():  # Disable gradient computation for validation
        for i, batch in enumerate(data_loader):
            loss = loss_fn(batch, model, criterion, device)
            running_loss += loss.item()
            
            pbar.update(data_loader.batch_size)
            pbar.set_postfix(loss=f"{running_loss / (i+1):.4f}")
    
    pbar.close()
    model.train()  # Set back to training mode

    avg_loss = running_loss / num_batches
    logger.info(f"Validation completed. Average loss: {avg_loss:.4f}")
    return avg_loss

def main(): 
    """Main training function."""
    # Set random seed for reproducibility
    set_seed(CONFIG["seed"])

    # Determine device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Prepare data loaders
    train_loader, valid_loader = prepare_dataloader(
        CONFIG["eeg_data_path"], 
        CONFIG["text_embedding_path"], 
        CONFIG["batch_size"], 
        CONFIG["num_workers"]
    )

    # Initialize model
    model = EEG_Embedding().to(device)
    logger.info(f"Model moved to {device}")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=CONFIG["epochs"] * len(train_loader)
    )
    criterion = F.mse_loss
    logger.info("Optimizer, scheduler, and loss function initialized")

    # Training tracking variables
    best_loss = float('inf')  # Track best validation loss instead of accuracy
    best_state_dict = None

    # Create checkpoint directory if it doesn't exist
    os.makedirs(CONFIG["save_checkpoint_path"], exist_ok=True)
    logger.info(f"Checkpoint directory: {CONFIG['save_checkpoint_path']}")

    # Training loop
    logger.info("Starting training...")
    pbar = tqdm(range(CONFIG['epochs']), desc='Train', unit='epoch', dynamic_ncols=True)
    
    for epoch in pbar:
        model.train()
        running_loss = 0.0
        
        # Training phase
        for i, batch in enumerate(train_loader):
            # Compute loss
            loss = loss_fn(batch, model, criterion, device)
            batch_loss = loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += batch_loss

        # Calculate average training loss for this epoch
        train_loss = running_loss / len(train_loader)
        pbar.set_postfix(
            train_loss=f"{train_loss:.4f}", 
            lr=f"{scheduler.get_last_lr()[0]:.2e}"
        )

        # Validation phase
        if (epoch + 1) % CONFIG['valid_steps'] == 0:
            valid_loss = validate_model(valid_loader, model, criterion, device)
            logger.info(f"Epoch {epoch+1}/{CONFIG['epochs']} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
            
            # Save best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state_dict = model.state_dict()
                logger.info(f"✅ Best validation loss updated: {best_loss:.4f}")

    pbar.close()

    # Save the best model
    if best_state_dict is not None:
        best_model_path = os.path.join(CONFIG["save_checkpoint_path"], "best_model.pth")
        torch.save(best_state_dict, best_model_path)
        logger.info(f"🎉 Best model saved to {best_model_path}")
        logger.info(f"Final best validation loss: {best_loss:.4f}")
    else:
        logger.warning("No best model was saved (validation was not performed)")

if __name__ == '__main__':
    main()
