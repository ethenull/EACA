# train_cahme_fixed.py
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
            num_workers=0
        )
        
        return dataloader
    
    def prepare_cahme_input(self, features):
        """Prepare input for CAHME model which expects separate text and audio features"""
        batch_size = features.shape[0]
        
        # Split multimodal features back into text and audio
        text_features = features[:, :768]  # First 768 dimensions
        audio_features = features[:, 768:768+168]  # Next 168 dimensions
        
        # Create the input structure CAHME expects
        current_features = {
            'text': text_features,
            'audio': audio_features,
            'multimodal': features
        }
        
        # For CAHME, we don't use history in this simple version
        history_features = None
        
        return current_features, history_features
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Prepare input based on model architecture
            if self.config.MODEL_ARCHITECTURE == 'm3fnet':
                # M3FNet expects sequence input
                features = features.unsqueeze(1)
                outputs = self.model(multimodal_features=features)
            else:  # CAHME
                # CAHME expects separate text/audio features
                current_features, history_features = self.prepare_cahme_input(features)
                outputs = self.model(current_features=current_features, history_features=history_features)
            
            # Forward pass
            self.optimizer.zero_grad()
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
                
                # Prepare input based on model architecture
                if self.config.MODEL_ARCHITECTURE == 'm3fnet':
                    features = features.unsqueeze(1)
                    outputs = self.model(multimodal_features=features)
                else:  # CAHME
                    current_features, history_features = self.prepare_cahme_input(features)
                    outputs = self.model(current_features=current_features, history_features=history_features)
                
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
    
    def save_model(self, filename):
        """Save model checkpoint - PyTorch 2.6 compatible"""
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
        print(f"💾 Model saved as {filename}")
    
    def load_model(self, filename):
        """Load model checkpoint - PyTorch 2.6 compatible"""
        try:
            checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
        except:
            try:
                checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
                print("⚠️  Loaded with weights_only=False")
            except Exception as e:
                print(f"❌ Loading failed: {e}")
                return False
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"📥 Model loaded from {filename}")
        return True
    
    def train(self):
        """Main training loop"""
        print(f"🚀 Starting training for {self.config.NUM_EPOCHS} epochs...")
        print(f"📊 Model: {self.config.MODEL_ARCHITECTURE.upper()}")
        print(f"📈 Dataset sizes - Train: {len(self.train_loader.dataset)}, Dev: {len(self.dev_loader.dataset)}")
        
        best_dev_acc = 0
        patience_counter = 0
        patience = 10
        
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
                self.save_model('best_model_cahme.pth')
                patience_counter = 0
                print(f"🎉 New best model saved! Dev Accuracy: {dev_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break
        
        # Final evaluation without loading (use current model)
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        # Test set evaluation
        test_loss, test_acc, test_pred, test_true = self.evaluate(self.test_loader)
        print(f"📊 Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(
            test_true, test_pred, 
            target_names=self.config.EMOTION_LABELS,
            digits=4
        ))
        
        return best_dev_acc, test_acc

def main():
    config = Config()
    trainer = EmotionTrainer(config)
    best_dev_acc, test_acc = trainer.train()
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Best Dev Accuracy: {best_dev_acc:.2f}%")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Model: {config.MODEL_ARCHITECTURE.upper()}")

if __name__ == "__main__":
    main()