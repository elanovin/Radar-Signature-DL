import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.data.dataset import RadarDataset
from src.models.model import RadarClassifier
from src.utils.trainer import ModelTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train radar classification model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                      help='Path to configuration file')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset and dataloaders
    train_dataset = RadarDataset(config['data']['train_path'], transform=True)
    val_dataset = RadarDataset(config['data']['val_path'], transform=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                          shuffle=False, num_workers=4)
    
    # Initialize model
    model = RadarClassifier(
        input_channels=config['model']['input_channels'],
        num_classes=config['model']['num_classes']
    )
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        config=config['training'],
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Train model
    trainer.train()
    
    # Save final model
    trainer.save_model('models/final_model.pth')

if __name__ == '__main__':
    main() 