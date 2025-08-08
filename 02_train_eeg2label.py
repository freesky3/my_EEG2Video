"""
EEG to Label Classification Training Script

This script trains a classification model that maps EEG signals to video labels.
The model uses the glfnet architecture to analyze both global and local features
in EEG data for video classification tasks.

Author: Deep Learning Research Team
Date: 2025
"""

import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from einops import rearrange, repeat
from tqdm import tqdm

# Import the model from the correct location
from models.eeg2label import glfnet

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eeg2label_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "data_path": "data/PSD_DE/imaging",  # Fixed path separator
    "label_path": "data/metadata/GT_label.npy",  # Fixed path separator
    "save_path": "checkpoints/eeg2label",

    "seed": 42, 
    "train_valid": [0.8, 0.2], 
    "batch_size": 32,
    "num_workers": 0,

    "input_dim": 62*5,  # 62 electrodes * 5 frequency bands
    "emb_dim": 64,
    "out_dim": 50,      # 50 video classes

    "learning_rate": 0.001, 
    "valid_steps": 10,
    "epochs": 100, 
}

def set_seed(seed):
    """Sets random seeds for reproducibility across all libraries.
    
    Args:
        seed (int): The seed to use for all random number generators.
    """
    logger.info(f"Setting random seed to {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class EEGLabelDataset(Dataset): 
    """PyTorch Dataset for EEG data and corresponding video labels.
    
    This dataset handles loading and preprocessing of EEG PSD/DE features
    and their corresponding video labels for classification tasks.
    """
    def __init__(self, data_path, label_path):
        """Initialize the dataset by loading and preprocessing data.
        
        Args:
            data_path (str): Path to the directory containing EEG PSD/DE files
            label_path (str): Path to the ground truth label file
        """
        self.data, self.label = self._load_data_and_labels(data_path, label_path)
        logger.info(f"Dataset initialized with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert to appropriate data types: float32 for data, int64 for labels
        # Subtract 1 from labels to make them 0-indexed (assuming original labels are 1-indexed)
        return self.data[idx].astype(np.float32), self.label[idx].astype(np.int64) - 1

    def _load_data_and_labels(self, data_path, label_path):
        """Load and preprocess EEG data and labels.
        
        Args:
            data_path (str): Path to EEG data directory
            label_path (str): Path to label file
            
        Returns:
            tuple: (processed_data, processed_labels)
        """
        logger.info("Loading EEG data and labels...")
        
        # Load EEG data from all files in the directory
        data_files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
        logger.info(f"Found {len(data_files)} EEG data files")
        
        data = []
        for file in data_files:
            file_path = os.path.join(data_path, file)
            file_data = np.load(file_path)
            data.append(file_data)
            logger.debug(f"Loaded {file} with shape: {file_data.shape}")
        
        data = np.stack(data, axis=0)
        logger.info(f"Stacked EEG data shape: {data.shape}")  # Expected: (n_subjects, 2, 5, 50, 62, 5)
        
        # Load ground truth labels
        labels = np.load(label_path)
        logger.info(f"Loaded labels with shape: {labels.shape}")  # Expected: (5, 50)
        
        # Reshape data: (n_subjects, 2, 5, 50, 62, 5) -> (n_subjects*2*5*50, 62*5)
        rearranged_data = rearrange(data, 'a b c d e f -> (a b c d) (e f)')
        logger.info(f"Reshaped data to: {rearranged_data.shape}")
        
        # Repeat labels to match data samples: (5, 50) -> (n_subjects*2*5*50,)
        rearranged_labels = repeat(labels, 'a b -> (repeat a b)', repeat=data.shape[0]*data.shape[1])
        logger.info(f"Reshaped labels to: {rearranged_labels.shape}")

        return rearranged_data, rearranged_labels

def get_dataloader(data_path, label_path, batch_size, num_workers=0):
    """Create training and validation DataLoaders.
    
    Args:
        data_path (str): Path to EEG data directory
        label_path (str): Path to label file
        batch_size (int): Batch size for DataLoaders
        num_workers (int): Number of worker processes
        
    Returns:
        tuple: (train_loader, valid_loader)
    """
    logger.info("Creating DataLoaders...")
    
    # Create dataset
    dataset = EEGLabelDataset(data_path, label_path)
    
    # Split into train and validation sets
    trainset, validset = random_split(dataset, CONFIG["train_valid"])
    logger.info(f"Split dataset: {len(trainset)} training, {len(validset)} validation samples")

    # Create DataLoaders
    train_loader = DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()  # Only use pin_memory if CUDA is available
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

def compute_loss_and_accuracy(batch, model, criterion, device):
    """Compute loss and accuracy for a batch of data.
    
    Args:
        batch: Tuple of (input_data, labels) from DataLoader
        model: The classification model
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: Device to run computation on
        
    Returns:
        tuple: (loss, accuracy) as torch tensors
    """
    input_data, labels = batch
    input_data = input_data.to(device)
    labels = labels.to(device)

    # Forward pass
    output = model(input_data)

    # Compute loss
    loss = criterion(output, labels)

    # Compute accuracy
    predictions = output.argmax(dim=1)
    accuracy = torch.mean((predictions == labels).float())

    return loss, accuracy

def validate_model(data_loader, model, criterion, device): 
    """Validate the model on the validation set.
    
    Args:
        data_loader: Validation DataLoader
        model: The model to validate
        criterion: Loss function
        device: Device to run computation on
        
    Returns:
        float: Average validation accuracy
    """
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    num_batches = len(data_loader)
    
    logger.info("Starting validation...")
    pbar = tqdm(total=len(data_loader.dataset), ncols=0, desc='Valid')

    with torch.no_grad():  # Disable gradient computation for validation
        for i, batch in enumerate(data_loader):
            loss, accuracy = compute_loss_and_accuracy(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()
            
            pbar.update(data_loader.batch_size)
            pbar.set_postfix(
                loss=f"{running_loss / (i+1):.4f}",
                accuracy=f"{running_accuracy / (i+1):.4f}",
            )
    
    pbar.close()
    model.train()  # Set back to training mode

    avg_accuracy = running_accuracy / num_batches
    avg_loss = running_loss / num_batches
    logger.info(f"Validation completed. Average accuracy: {avg_accuracy:.4f}, Average loss: {avg_loss:.4f}")
    return avg_accuracy

def main():
    """Main training function."""
    # Set random seed for reproducibility
    set_seed(CONFIG["seed"])

    # Determine device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create data loaders
    train_loader, valid_loader = get_dataloader(
        CONFIG["data_path"], 
        CONFIG["label_path"], 
        CONFIG["batch_size"], 
        num_workers=CONFIG["num_workers"]
    )
    logger.info("Data loading completed")

    # Initialize model
    model = glfnet(
        input_dim=CONFIG["input_dim"], 
        emb_dim=CONFIG["emb_dim"], 
        out_dim=CONFIG["out_dim"]
    ).to(device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Initialize loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=CONFIG["epochs"] * len(train_loader)  # Fixed: use CONFIG["epochs"] instead of hardcoded 200
    )
    logger.info("Optimizer, scheduler, and loss function initialized")

    # Training tracking variables
    best_accuracy = 0.0
    best_state_dict = None

    # Create checkpoint directory if it doesn't exist
    os.makedirs(CONFIG["save_path"], exist_ok=True)
    logger.info(f"Checkpoint directory: {CONFIG['save_path']}")

    # Training loop
    logger.info("Starting training...")
    pbar = tqdm(range(CONFIG['epochs']), desc='Train', unit='epoch', dynamic_ncols=True)
    
    for epoch in pbar:
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        
        # Training phase
        for i, batch in enumerate(train_loader):
            # Compute loss and accuracy
            loss, accuracy = compute_loss_and_accuracy(batch, model, criterion, device)
            batch_loss = loss.item()
            batch_accuracy = accuracy.item()

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients first
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += batch_loss
            running_accuracy += batch_accuracy

        # Calculate average training metrics for this epoch
        train_loss = running_loss / len(train_loader)
        train_acc = running_accuracy / len(train_loader)
        pbar.set_postfix(
            train_loss=f"{train_loss:.4f}", 
            train_acc=f"{train_acc:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}"
        )

        # Validation phase
        if (epoch + 1) % CONFIG['valid_steps'] == 0:
            valid_acc = validate_model(valid_loader, model, criterion, device)
            logger.info(f"Epoch {epoch+1}/{CONFIG['epochs']} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}")
            
            # Save best model
            if valid_acc > best_accuracy:
                best_accuracy = valid_acc
                best_state_dict = model.state_dict()
                logger.info(f"✅ Best validation accuracy updated: {best_accuracy:.4f}")

    pbar.close()

    # Save the best model
    if best_state_dict is not None:
        best_model_path = os.path.join(CONFIG["save_path"], "best_model.pth")
        torch.save(best_state_dict, best_model_path)
        logger.info(f"🎉 Best model saved to {best_model_path}")
        logger.info(f"Final best validation accuracy: {best_accuracy:.4f}")
    else:
        logger.warning("No best model was saved (validation was not performed)")

    return best_accuracy

if __name__ == '__main__':
    best_acc = main()
    print(f"\n[Final] Training finished with best validation accuracy: {best_acc:.4f}")
