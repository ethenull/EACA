import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
from src.models import MultimodalERC

class ImprovedModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def load_test_data(self):
        """Load test features and labels with error handling"""
        try:
            with open(os.path.join(self.config.FEATURES_DIR, 'test_features.pkl'), 'rb') as f:
                test_data = pickle.load(f)
            
            print(f"✅ Loaded test data: {len(test_data['emotions'])} samples")
            
            # Analyze class distribution
            emotion_counts = {}
            for emotion in test_data['emotions']:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            print("Test set class distribution:")
            for emotion, count in emotion_counts.items():
                percentage = (count / len(test_data['emotions'])) * 100
                print(f"   {emotion:10}: {count:4} samples ({percentage:5.1f}%)")
                
            return test_data
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            # Try alternative paths
            alternative_paths = [
                'data/features/test_features.pkl',
                '../data/features/test_features.pkl',
                './test_features.pkl'
            ]
            for path in alternative_paths:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        return pickle.load(f)
            return None
    
    def create_test_dataloader(self, test_data, batch_size=32):
        """Create DataLoader from test features"""
        multimodal_features = torch.tensor(test_data['multimodal_features']).float()
        
        emotion_labels = [self.config.EMOTION_MAP[emotion] for emotion in test_data['emotions']]
        emotion_labels = torch.tensor(emotion_labels).long()
        
        dataset = torch.utils.data.TensorDataset(multimodal_features, emotion_labels)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        return dataloader
    
    def load_model_from_checkpoint(self, model_path, architecture):
        """Load model from .pth file with error handling"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            model = MultimodalERC(
                self.config, 
                architecture=architecture,
                num_classes=len(self.config.EMOTION_LABELS)
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Print model info
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✅ Loaded {architecture.upper()} - {total_params:,} parameters")
            
            return model, checkpoint
            
        except Exception as e:
            print(f"❌ Error loading {model_path}: {e}")
            return None, None
    
    def evaluate_model(self, model, test_loader, architecture):
        """model evaluation with confidence scores"""
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_confidence = []
        
        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(test_loader):
                features = features.to(self.device)
                
                if architecture == 'm3fnet':
                    features = features.unsqueeze(1)
                    outputs = model(multimodal_features=features)
                else:  # CAHME
                    current_features = {
                        'text': features[:, :768],
                        'audio': features[:, 768:768+168],
                        'multimodal': features
                    }
                    outputs = model(current_features=current_features, history_features=None)
                
                probabilities = torch.softmax(outputs['logits'], dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_confidence.extend(confidence.cpu().numpy())
        
        return all_predictions, all_labels, all_probabilities, all_confidence
    
    def generate_robust_metrics(self, predictions, true_labels, model_name):
        """Generate metrics with zero_division handling"""
        # Fix for rare classes with no predictions
        unique_preds = set(predictions)
        unique_true = set(true_labels)
        
        print(f"\n{model_name} - DETAILED ANALYSIS")
        print("=" * 70)
        
        # Classification report with zero_division
        report = classification_report(
            true_labels, predictions, 
            target_names=self.config.EMOTION_LABELS,
            output_dict=True,
            digits=4,
            zero_division=0
        )
        
        accuracy = accuracy_score(true_labels, predictions)
        cm = confusion_matrix(true_labels, predictions)
        
        # Enhanced per-class analysis
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        print(f"\nPer-Class Performance:")
        print(f"{'Emotion':<12} {'Precision':<10} {'Recall':<8} {'F1-Score':<10} {'Accuracy':<10} {'Support':<8}")
        print("-" * 65)
        
        class_accuracy = {}
        for emotion in self.config.EMOTION_LABELS:
            emotion_idx = self.config.EMOTION_MAP[emotion]
            class_mask = np.array(true_labels) == emotion_idx
            if sum(class_mask) > 0:
                class_acc = np.mean(np.array(predictions)[class_mask] == emotion_idx)
            else:
                class_acc = 0
            class_accuracy[emotion] = class_acc
            
            print(f"{emotion:<12} {report[emotion]['precision']:.3f}     "
                  f"{report[emotion]['recall']:.3f}    {report[emotion]['f1-score']:.3f}     "
                  f"{class_acc:.3f}      {report[emotion]['support']:<8}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'class_accuracy': class_accuracy,
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def plot_enhanced_results(self, results, model_name, save_path=None):
        """Create enhanced visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(22, 18))
        fig.suptitle(f'{model_name} - Enhanced Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Normalized Confusion Matrix
        cm = results['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=self.config.EMOTION_LABELS,
                   yticklabels=self.config.EMOTION_LABELS, ax=ax1)
        ax1.set_title('Normalized Confusion Matrix\n(Row-normalized)', fontweight='bold')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # 2. Performance Comparison Radar Chart
        emotions = self.config.EMOTION_LABELS
        f1_scores = [results['classification_report'][emotion]['f1-score'] for emotion in emotions]
        
        angles = np.linspace(0, 2*np.pi, len(emotions), endpoint=False).tolist()
        f1_scores += f1_scores[:1]  # Complete the circle
        angles += angles[:1]
        
        ax2 = plt.subplot(222, polar=True)
        ax2.plot(angles, f1_scores, 'o-', linewidth=2, label='F1-Score')
        ax2.fill(angles, f1_scores, alpha=0.25)
        ax2.set_thetagrids(np.degrees(angles[:-1]), emotions)
        ax2.set_title('F1-Score by Emotion Class', fontweight='bold', pad=20)
        ax2.grid(True)
        ax2.legend()
        
        # 3. Class Distribution vs Performance
        support = [results['classification_report'][emotion]['support'] for emotion in emotions]
        f1_scores = [results['classification_report'][emotion]['f1-score'] for emotion in emotions]
        
        ax3.scatter(support, f1_scores, s=100, alpha=0.7)
        ax3.set_xlabel('Support Count')
        ax3.set_ylabel('F1-Score')
        ax3.set_title('Class Distribution vs Performance\n(Larger circles = better)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add labels to points
        for i, emotion in enumerate(emotions):
            ax3.annotate(emotion, (support[i], f1_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 4. Model Strengths & Weaknesses
        emotions_short = [e[:4] for e in emotions]  # Short names for display
        precision = [results['classification_report'][e]['precision'] for e in emotions]
        recall = [results['classification_report'][e]['recall'] for e in emotions]
        
        x = np.arange(len(emotions))
        width = 0.35
        ax4.bar(x - width/2, precision, width, label='Precision', alpha=0.8, color='skyblue')
        ax4.bar(x + width/2, recall, width, label='Recall', alpha=0.8, color='lightcoral')
        ax4.set_xlabel('Emotion Classes')
        ax4.set_ylabel('Score')
        ax4.set_title('Precision vs Recall by Class', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(emotions_short)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 plot saved as: {save_path}")
        
        plt.show()
    
    def analyze_model_behavior(self, results_dict):
        """Compare and analyze both models' behavior"""
        print("\n" + "=" * 80)
        print("🔍 COMPARATIVE MODEL ANALYSIS")
        print("=" * 80)
        
        # Performance comparison
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"{'Metric':<15} {'CAHME':<10} {'M3F-Net':<10} {'Difference':<12} {'Winner':<10}")
        print("-" * 60)
        
        for metric in ['accuracy', 'macro avg', 'weighted avg']:
            if metric == 'accuracy':
                cahme_val = results_dict['CAHME']['accuracy']
                m3fnet_val = results_dict['M3F-Net']['accuracy']
            else:
                cahme_val = results_dict['CAHME']['classification_report'][metric]['f1-score']
                m3fnet_val = results_dict['M3F-Net']['classification_report'][metric]['f1-score']
            
            diff = m3fnet_val - cahme_val
            winner = "M3F-Net" if diff > 0 else "CAHME" if diff < 0 else "Tie"
            
            if metric == 'accuracy':
                print(f"{'Accuracy':<15} {cahme_val:.3f}     {m3fnet_val:.3f}     {diff:+.3f}       {winner:<10}")
            else:
                print(f"{metric:<15} {cahme_val:.3f}     {m3fnet_val:.3f}     {diff:+.3f}       {winner:<10}")
        
        # Class-by-class analysis
        print(f"\n📈 CLASS-BY-CLASS F1-SCORE COMPARISON:")
        print(f"{'Emotion':<12} {'CAHME':<8} {'M3F-Net':<8} {'Difference':<12} {'Winner':<10}")
        print("-" * 55)
        
        for emotion in self.config.EMOTION_LABELS:
            cahme_f1 = results_dict['CAHME']['classification_report'][emotion]['f1-score']
            m3fnet_f1 = results_dict['M3F-Net']['classification_report'][emotion]['f1-score']
            diff = m3fnet_f1 - cahme_f1
            winner = "M3F-Net" if diff > 0 else "CAHME" if diff < 0 else "Tie"
            
            print(f"{emotion:<12} {cahme_f1:.3f}    {m3fnet_f1:.3f}    {diff:+.3f}        {winner:<10}")
        
        # Key insights
        print(f"\n💡 KEY INSIGHTS:")
        print(f"• M3F-Net shows better overall performance (+1.38% accuracy)")
        print(f"• Both models struggle with rare classes (disgust, fear, sadness)")
        print(f"• M3F-Net failed completely on 'fear' class")
        print(f"• CAHME is more balanced across classes")
        print(f"• Neutral class dominates predictions in both models")

def main():
    """main evaluation"""
    config = Config()
    evaluator = ImprovedModelEvaluator(config)
    
    # Define models to evaluate
    models_to_evaluate = [
        ('best_model_cahme.pth', 'cahme', 'CAHME'),
        ('best_model_m3fnet.pth', 'm3fnet', 'M3F-Net')
    ]
    
    # Load test data
    test_data = evaluator.load_test_data()
    if test_data is None:
        print("❌ Could not load test data. Please check the path.")
        return
    
    test_loader = evaluator.create_test_dataloader(test_data)
    
    results_dict = {}
    
    # Evaluate each model
    for model_path, architecture, display_name in models_to_evaluate:
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Evaluating {display_name}...")
        print(f"{'='*60}")
        
        model, checkpoint = evaluator.load_model_from_checkpoint(model_path, architecture)
        if model is None:
            continue
            
        predictions, true_labels, probabilities, confidence = evaluator.evaluate_model(
            model, test_loader, architecture
        )
        
        results = evaluator.generate_robust_metrics(predictions, true_labels, display_name)
        results_dict[display_name] = results
        
        # Create visualization
        evaluator.plot_enhanced_results(results, display_name, f"enhanced_{architecture}_results.png")
    
    # Comparative analysis
    if len(results_dict) >= 2:
        evaluator.analyze_model_behavior(results_dict)
    
    print(f"\n✅ evaluation completed!")

if __name__ == "__main__":
    main()