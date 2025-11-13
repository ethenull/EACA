# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from config import Config
from src.models import MultimodalERC

class EmotionTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load features
        self.train_features = self.load_features('train')
        self.dev_features = self.load_features('dev') 
        self.test_features = self.load_features('test')
        
        # Prepare datasets
        self.train_loader = self.prepare_dataloader(self.train_features, shuffle=True)
        self.dev_loader = self.prepare_dataloader(self.dev_features, shuffle=False)
        self.test_loader = self.prepare_dataloader(self.test_features, shuffle=False)
        
        # Initialize model
        self.model = MultimodalERC(
            config, 
            architecture=config.MODEL_ARCHITECTURE,
            num_classes=len(config.EMOTION_LABELS)
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'dev_loss': [], 'dev_acc': [],
            'learning_rates': []
        }
    
    def load_features(self, split):
        """Load preprocessed features"""
        features_path = os.path.join(self.config.FEATURES_DIR, f"{split}_features.pkl")
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
        print(f"✅ Loaded {split} features: {features['multimodal_features'].shape}")
        return features
    
    def prepare_dataloader(self, features, shuffle=False):
        """Create DataLoader from features"""
        # Convert to tensors
        multimodal_features = torch.tensor(features['multimodal_features']).float()
        
        # Convert emotion labels to indices
        emotion_labels = [self.config.EMOTION_MAP[emotion] for emotion in features['emotions']]
        emotion_labels = torch.tensor(emotion_labels).long()
        
        # Create dataset
        dataset = TensorDataset(multimodal_features, emotion_labels)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        return dataloader
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Add sequence dimension for M3F-Net
            if self.config.MODEL_ARCHITECTURE == 'm3fnet':
                features = features.unsqueeze(1)  # (batch, 1, features)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(multimodal_features=features)
            loss = self.criterion(outputs['logits'], labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs['logits'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader):
        """Evaluate model on given dataloader"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Add sequence dimension for M3F-Net
                if self.config.MODEL_ARCHITECTURE == 'm3fnet':
                    features = features.unsqueeze(1)
                
                outputs = self.model(multimodal_features=features)
                loss = self.criterion(outputs['logits'], labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs['logits'], 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self):
        """Main training loop"""
        print(f"🚀 Starting training for {self.config.NUM_EPOCHS} epochs...")
        print(f"📊 Model: {self.config.MODEL_ARCHITECTURE.upper()}")
        print(f"📈 Dataset sizes - Train: {len(self.train_loader.dataset)}, Dev: {len(self.dev_loader.dataset)}")
        
        best_dev_acc = 0
        patience_counter = 0
        patience = 10  # Early stopping patience
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Evaluate
            dev_loss, dev_acc, _, _ = self.evaluate(self.dev_loader)
            
            # Update learning rate
            self.scheduler.step(dev_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['dev_loss'].append(dev_loss)
            self.history['dev_acc'].append(dev_acc)
            self.history['learning_rates'].append(current_lr)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                self.save_model('best_model.pth')
                patience_counter = 0
                print(f"🎉 New best model saved! Dev Accuracy: {dev_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model for final evaluation
        self.load_model('best_model.pth')
        
        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        # Test set evaluation
        test_loss, test_acc, test_pred, test_true = self.evaluate(self.test_loader)
        print(f"📊 Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
        
        # Detailed classification report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(
            test_true, test_pred, 
            target_names=self.config.EMOTION_LABELS,
            digits=4
        ))
        
        # Plot training history
        self.plot_training_history()
        
        return best_dev_acc, test_acc
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': len(self.history['train_loss']),
            'best_dev_acc': max(self.history['dev_acc']) if self.history['dev_acc'] else 0,
            'config': {
                'MODEL_ARCHITECTURE': self.config.MODEL_ARCHITECTURE,
                'NUM_CLASSES': len(self.config.EMOTION_LABELS)
            }
        }
        torch.save(checkpoint, filename)
        print(f"💾 Model saved as {filename} (Dev Acc: {checkpoint['best_dev_acc']:.2f}%)")
    
    def load_model(self, filename):
        """Load model checkpoint with PyTorch 2.6 compatibility"""
        try:
            # First try with weights_only=True (PyTorch 2.6+ default)
            checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
        except:
            try:
                # If that fails, try the old way (less secure but works)
                checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
                print("⚠️  Loaded with weights_only=False - only use trusted models")
            except Exception as e:
                # If both fail, try a more compatible approach
                print(f"⚠️  Standard loading failed: {e}")
                print("🔄 Trying alternative loading method...")
                
                # Load just the model state dict
                checkpoint = torch.load(filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Only load optimizer if it exists in checkpoint
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"📥 Model loaded from {filename}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['dev_loss'], label='Dev Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['dev_acc'], label='Dev Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate
        ax3.plot(self.history['learning_rates'])
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)
        
        # Confusion matrix (simplified)
        _, _, test_pred, test_true = self.evaluate(self.test_loader)
        cm = confusion_matrix(test_true, test_pred)
        im = ax4.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax4.set_title('Confusion Matrix')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('True')
        ax4.set_xticks(range(len(self.config.EMOTION_LABELS)))
        ax4.set_yticks(range(len(self.config.EMOTION_LABELS)))
        ax4.set_xticklabels(self.config.EMOTION_LABELS, rotation=45)
        ax4.set_yticklabels(self.config.EMOTION_LABELS)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    config = Config()
    trainer = EmotionTrainer(config)
    best_dev_acc, test_acc = trainer.train()
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Best Dev Accuracy: {best_dev_acc:.2f}%")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Model: {config.MODEL_ARCHITECTURE.upper()}")
    print(f"   Model saved as: best_model.pth")
    print(f"   Training plot saved as: training_history.png")

if __name__ == "__main__":
    main()