# main.py
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

from src.preprocessor import TextPreprocessor, AudioPreprocessor, DataPreprocessor
from src.feature_extractor import MultimodalFeatureExtractor
from src.data_loader import MELDDataset
from config import Config
import torch

from src.models import MultimodalERC, ContextAwareHierarchicalMultimodalEncoder
from src.trainer import MultimodalTrainer
#from src.data_collator import MeldCollator

Config = Config()

def test_text_preprocessing():
    """Test text preprocessing functionality"""
    print("=== Testing Text Preprocessing ===")
    test_text = "These are CONTRACTIONS DON'T I'M Y'ALL WHERE'D lower case time who's john's ain't.#()@)($@*()$%*(_+++!_#$|\\]}))"
    cleaned_text = TextPreprocessor.clean_text(test_text)
    tokenized_text = TextPreprocessor.tokenize(cleaned_text)
    
    print(f"Original text: {test_text}")
    print(f"Cleaned text: {cleaned_text}")
    print(f"Tokenized text: {tokenized_text}")
    print()

def test_audio_preprocessing():
    """Test audio preprocessing functionality"""
    print("=== Testing Audio Preprocessing ===")
    
    # Use MELD dataset
    test_audio_path = os.path.join(Config.TRAIN_VIDEO_DIR, "dia0_utt0.mp4")
    
    if os.path.exists(test_audio_path):
        try:
            # Test loading audio
            result = AudioPreprocessor.load_audio(test_audio_path, 16000)
            if result:
                audio, sr = result
                print(f"Loaded audio - Shape: {audio.shape}, Sample rate: {sr}")
                
                # Test cleaning audio
                cleaned_audio = AudioPreprocessor.clean_audio(audio, 20)
                print(f"Cleaned audio - Shape: {cleaned_audio.shape}")
                
                # Test full preprocessing
                processed_result = AudioPreprocessor.preprocess_audio_given_path(test_audio_path, 16000, 20)
                if processed_result:
                    audio_data, sample_rate = processed_result
                    print(f"Fully processed audio - Shape: {audio_data.shape}, Sample rate: {sample_rate}")
                else:
                    print("Full audio preprocessing failed")
            else:
                print("Audio loading returned None")
            
        except Exception as e:
            print(f"Audio processing error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Test audio file not found: {test_audio_path}")
        print("Generating synthetic audio for testing...")
        
        # Test with synthetic audio
        synthetic_audio, sr = AudioPreprocessor.generate_synthetic_audio()
        print(f"Synthetic audio - Shape: {synthetic_audio.shape}, Sample rate: {sr}")
        
        cleaned_audio = AudioPreprocessor.clean_audio(synthetic_audio, 20)
        print(f"Cleaned synthetic audio - Shape: {cleaned_audio.shape}")
    
    print()

def test_feature_extraction():
    """Test feature extraction functionality"""
    print("=== Testing Feature Extraction ===")
    
    try:
        # Create multimodal feature extractor
        feature_extractor = MultimodalFeatureExtractor()
        
        # Test text with sample data
        test_sample = {
            'tokenized_text': "this is a test utterance for feature extraction",
            'processed_audio': None,  # No audio for this test
            'text': "this is a test utterance for feature extraction",
            'audio_path': None
        }
        
        # Extract features without dialogue history
        features = feature_extractor.extract_features(test_sample)
        
        print(f"Text features shape: {features['text'].shape}")
        print(f"Audio features shape: {features['audio'].shape}")
        print(f"Multimodal features shape: {features['multimodal'].shape}")
        print(f"Text features type: {type(features['text'])}")
        print(f"Audio features type: {type(features['audio'])}")
        print()
        
        # Test with synthetic audio
        print("Testing with synthetic audio...")
        try:
            import numpy as np
            
            # Create synthetic audio
            sr = 16000
            t = np.linspace(0, 1, sr)
            synthetic_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
            
            test_sample_with_audio = {
                'tokenized_text': "test with audio features",
                'processed_audio': (synthetic_audio, sr),
                'text': "test with audio features",
                'audio_path': "synthetic_audio.wav"
            }
            
            features_with_audio = feature_extractor.extract_features(test_sample_with_audio)
            print(f"With audio - Text features shape: {features_with_audio['text'].shape}")
            print(f"With audio - Audio features shape: {features_with_audio['audio'].shape}")
            print(f"With audio - Multimodal features shape: {features_with_audio['multimodal'].shape}")
            
        except Exception as e:
            print(f"Audio feature extraction test failed: {e}")
            
    except Exception as e:
        print(f"Feature extraction test failed: {e}")
    print()

def test_data_loader():
    """Test data loader functionality"""
    print("=== Testing Data Loader ===")
    
    try:
        # Try to load a small dataset sample
        dataset = MELDDataset(Config.DATA_DIR, split='train')
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            # Get first sample
            sample = dataset[0]  # Use __getitem__ directly
            print("Sample keys:", sample.keys())
            print("Text:", sample['text'][:50] + "..." if len(sample['text']) > 50 else sample['text'])
            print("Emotion:", sample['emotion'])
            print("Sentiment:", sample['sentiment'])
            print("Audio path:", sample['audio_path'])
            
            # Test a few more samples
            print(f"\nTesting first 3 samples:")
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                print(f"Sample {i}: Text='{sample['text'][:30]}...', Emotion={sample['emotion']}, Audio={sample['audio_path'] is not None}")
                
        else:
            print("Dataset is empty - check data paths")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure the MELD dataset is properly set up in the data directory")
        
        # Show what's in the data directory for debugging
        print(f"\nContents of {Config.DATA_DIR}:")
        if os.path.exists(Config.DATA_DIR):
            for item in os.listdir(Config.DATA_DIR):
                print(f"  {item}")
        else:
            print(f"Directory {Config.DATA_DIR} does not exist")
    print()

def test_full_pipeline():
    """Test the complete preprocessing and feature extraction pipeline with detailed output"""
    print("=== Testing Full Pipeline ===")
    
    try:
        # Load a sample from dataset
        dataset = MELDDataset(Config.DATA_DIR, split='train')
        if len(dataset) > 0:
            sample = dataset[0]
            print(f" Original sample text: {sample['text']}")
            print(f" Original audio path: {sample['audio_path']}")
            print(f" Emotion: {sample['emotion']}")
            print(f" Sentiment: {sample['sentiment']}")
            print(f" Speaker: {sample['speaker']}")
            print(f" Dialogue ID: {sample['dialogue_id']}")
            print(f"️ Utterance ID: {sample['utterance_id']}")
            
            # Preprocess the sample
            print("\n Preprocessing sample...")
            preprocessed_sample = DataPreprocessor.preprocess_sample(sample)
            
            print(f" Cleaned text: {preprocessed_sample.get('cleaned_text', 'Not found')}")
            print(f" Tokenized text: {preprocessed_sample.get('tokenized_text', 'Not found')}")
            
            # Check audio processing details
            processed_audio = preprocessed_sample.get('processed_audio')
            if processed_audio:
                audio_data, sample_rate = processed_audio
                print(f" Processed audio - Shape: {audio_data.shape}, Sample rate: {sample_rate}")
                print(f"️ Audio duration: {len(audio_data)/sample_rate:.2f} seconds")
            else:
                print(" No audio processed")
            
            # Extract features
            print("\n Extracting features...")
            feature_extractor = MultimodalFeatureExtractor()
            features = feature_extractor.extract_features(preprocessed_sample)
            
            print(" Feature extraction successful!")
            print(f" Text feature vector: {len(features['text'])} dimensions")
            print(f" Audio feature vector: {len(features['audio'])} dimensions") 
            print(f" Multimodal feature vector: {len(features['multimodal'])} dimensions")
            
            # Show first few values of each feature type
            print(f" Text features (first 5): {features['text'][:5]}")
            print(f" Audio features (first 5): {features['audio'][:5]}")
            print(f" Multimodal features (first 5): {features['multimodal'][:5]}")
            
            # Test multiple samples with detailed output
            print(f"\n Testing pipeline on first 3 samples:")
            successful = 0
            for i in range(min(3, len(dataset))):
                try:
                    sample = dataset[i]
                    print(f"\n--- Sample {i} ---")
                    print(f"Text: '{sample['text'][:50]}...'")
                    print(f"Emotion: {sample['emotion']}")
                    
                    preprocessed = DataPreprocessor.preprocess_sample(sample)
                    features = feature_extractor.extract_features(preprocessed)
                    
                    if features['multimodal'] is not None and len(features['multimodal']) > 0:
                        successful += 1
                        print(f" Success - Features: {len(features['multimodal'])}D")
                    else:
                        print(f" Failed - empty features")
                        
                except Exception as e:
                    print(f" Failed - {e}")
            
            print(f"\n Success rate: {successful}/{min(3, len(dataset))}")
                    
        else:
            print("No samples in dataset to test pipeline")
            
    except Exception as e:
        print(f"Error in full pipeline test: {e}")
        import traceback
        traceback.print_exc()
    print()

def test_config():
    """Test configuration settings"""
    print("=== Testing Configuration ===")
    print(f"Data directory: {Config.DATA_DIR}")
    print(f"Train CSV: {Config.TRAIN_CSV}")
    print(f"Emotion labels: {Config.EMOTION_LABELS}")
    print(f"Sentiment labels: {Config.SENTIMENT_LABELS}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Learning rate: {Config.LEARNING_RATE}")
    print()

def test_model_architectures():
    """Test both model architectures"""
    print("=== Testing Model Architectures ===")
    
    # Test CAHME
    print("\n Testing CAHME Architecture:")
    cahme_model = MultimodalERC(Config, architecture='cahme', num_classes=len(Config.EMOTION_LABELS))
    
    # Create dummy input
    batch_size = 2
    dummy_current = {
        'text': torch.randn(batch_size, 768),
        'audio': torch.randn(batch_size, 168),
        'multimodal': torch.randn(batch_size, 936)
    }
    
    try:
        output = cahme_model(current_features=dummy_current)
        print(f" CAHME forward pass successful")
        print(f"   Logits shape: {output['logits'].shape}")
        print(f"   Gate values: {output.get('gate_values', 'N/A')}")
    except Exception as e:
        print(f" CAHME test failed: {e}")

def main():
    """Main function to test all components"""
    print("Starting comprehensive test of the multimodal emotion recognition system...\n")
    
    # Test configuration first
    test_config()
    
    # Test individual components
    test_text_preprocessing()
    test_audio_preprocessing()
    test_feature_extraction()
    test_data_loader()
    
    # Test integrated pipeline
    test_full_pipeline()
    
    # New tests
    test_model_architectures()
    
    print("=== Testing Complete ===")
    print("All components have been tested. Check output above for any errors.")
    
if __name__ == "__main__":
    main()