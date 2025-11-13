# enhanced_train.py
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
        
        # Initialize model
        self.model = MultimodalERC(
            config, 
            architecture=config.MODEL_ARCHITECTURE,
            num_classes=len(config.EMOTION_LABELS)
        ).to(self.device)
        
        # Handle class imbalance - weighted loss based on your distribution
        class_weights = torch.tensor([
            1.0,  # anger (1000) 
            7.0,  # disgust (500) - highest weight
            7.0,  # fear (500) - highest weight  
            3.5,  # joy (2000)
            1.0,  # neutral (7000) - lowest weight
            3.5,  # sadness (1000)
            3.5   # surprise (1000)
        ]).float().to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=0.01
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
    def load_data(self, split):
        """Load preprocessed features"""
        features_path = os.path.join(self.config.FEATURES_DIR, f"{split}_features.pkl")
        
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
        
        # Convert to tensors
        multimodal_features = torch.tensor(features['multimodal_features']).float()
        
        # Convert emotion labels to indices
        emotion_indices = [self.config.EMOTION_MAP[emotion] for emotion in features['emotions']]
        labels = torch.tensor(emotion_indices).long()
        
        return TensorDataset(multimodal_features, labels), features
    
    def prepare_model_input(self, features, architecture='cahme'):
        """Prepare input features based on model architecture"""
        batch_size = features.shape[0]
        
        if architecture.lower() == 'cahme':
            # For CAHME: split into text and audio features
            text_features = features[:, :768]      # First 768D for text
            audio_features = features[:, 768:936]  # Next 168D for audio
            
            # Create input dictionary for CAHME
            model_input = {
                'current_features': {
                    'text': text_features,
                    'audio': audio_features,
                    'multimodal': features
                }
            }
            
        elif architecture.lower() == 'm3fnet':
            # For M3F-Net: just pass multimodal features directly
            model_input = {
                'multimodal_features': features.unsqueeze(1)  # Add sequence dimension
            }
        
        return model_input
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Prepare input based on model architecture
            model_input = self.prepare_model_input(features, self.config.MODEL_ARCHITECTURE)
            
            # Forward pass
            outputs = self.model(**model_input)
            loss = self.criterion(outputs['logits'], labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs['logits'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Prepare input based on model architecture
                model_input = self.prepare_model_input(features, self.config.MODEL_ARCHITECTURE)
                
                outputs = self.model(**model_input)
                loss = self.criterion(outputs['logits'], labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs['logits'], 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self):
        """Main training loop"""
        print("Loading training data...")
        train_dataset, train_features = self.load_data('train')
        val_dataset, val_features = self.load_data('dev')
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Model architecture: {self.config.MODEL_ARCHITECTURE}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False,
            num_workers=0
        )
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        print("Starting training...")
        for epoch in range(self.config.NUM_EPOCHS):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{self.config.NUM_EPOCHS}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.2e}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best_model.pth')
                print(f'  New best model saved!')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model for final evaluation
        self.load_model('best_model.pth')
        
        # Final evaluation
        print("\nFinal Evaluation:")
        self.evaluate(val_loader, split='Validation')
        
        # Test evaluation
        test_dataset, _ = self.load_data('test')
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        self.evaluate(test_loader, split='Test')
    
    def evaluate(self, data_loader, split='Test'):
        """Comprehensive evaluation"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Prepare input based on model architecture
                model_input = self.prepare_model_input(features, self.config.MODEL_ARCHITECTURE)
                
                outputs = self.model(**model_input)
                probabilities = torch.softmax(outputs['logits'], dim=1)
                _, predicted = torch.max(outputs['logits'], 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = (all_predictions == all_labels).mean()
        
        print(f"\n{split} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, 
                                  target_names=self.config.EMOTION_LABELS,
                                  digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Per-class accuracy
        print("\nPer-class Accuracy:")
        for i, emotion in enumerate(self.config.EMOTION_LABELS):
            class_mask = all_labels == i
            if class_mask.sum() > 0:
                class_acc = (all_predictions[class_mask] == all_labels[class_mask]).mean()
                print(f"  {emotion}: {class_acc:.4f} ({class_mask.sum()} samples)")
    
    def save_model(self, path):
        """Save model checkpoint - simplified version"""
        torch.save(self.model.state_dict(), path)
        print(f"Model state dict saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint - simplified version"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model state dict loaded from {path}")

def main():
    config = Config()
    trainer = EmotionTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()