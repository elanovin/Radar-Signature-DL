import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
import logging
from pathlib import Path
import time

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate']
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logging.info(f'Training batch {batch_idx}/{len(self.train_loader)} '
                           f'Loss: {loss.item():.6f}')
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        val_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        val_loss /= len(self.val_loader)
        accuracy = correct / len(self.val_loader.dataset)
        
        logging.info(f'Validation Loss: {val_loss:.6f}, '
                    f'Accuracy: {accuracy:.4f}')
        
        return val_loss
    
    def train(self):
        """Main training loop."""
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            logging.info(f'Epoch {epoch+1}/{self.config["epochs"]}')
            
            # Train and validate
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('models/best_model.pth')
            
            logging.info(f'Epoch {epoch+1} - '
                        f'Training Loss: {train_loss:.6f}, '
                        f'Validation Loss: {val_loss:.6f}')
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logging.info(f'Model saved to {path}') 