# config.py
import os

class Config:
    # Data paths
    DATA_DIR = "data/meld"
    RAW_DATA_DIR = "data/meld/raw"
    PROCESSED_DATA_DIR = "data/processed"
    FEATURES_DIR = "data/features"
    
    # CSV files
    TRAIN_CSV = os.path.join(DATA_DIR, "train_sent_emo.csv")
    DEV_CSV = os.path.join(DATA_DIR, "dev_sent_emo.csv")
    TEST_CSV = os.path.join(DATA_DIR, "test_sent_emo.csv")
    
    # Video directories
    TRAIN_VIDEO_DIR = os.path.join(DATA_DIR, "train_splits")
    DEV_VIDEO_DIR = os.path.join(DATA_DIR, "dev_splits_complete")
    TEST_VIDEO_DIR = os.path.join(DATA_DIR, "output_repeated_splits_test")
    
    # Feature directories
    TEXT_FEATURES_DIR = os.path.join(FEATURES_DIR, "text")
    AUDIO_FEATURES_DIR = os.path.join(FEATURES_DIR, "audio")
    MULTIMODAL_FEATURES_DIR = os.path.join(FEATURES_DIR, "multimodal")
    
    # Model parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    
    # Feature extraction
    TEXT_EMBEDDING_DIM = 768
    AUDIO_FEATURE_DIM = 168
    MULTIMODAL_FEATURE_DIM = 936  # 768 + 168
    
    # Audio parameters
    TARGET_SAMPLE_RATE = 16000
    N_MFCC = 40
    
    # Emotion labels
    EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    EMOTION_MAP = {emotion: idx for idx, emotion in enumerate(EMOTION_LABELS)}
    
    # Sentiment labels
    SENTIMENT_LABELS = ['positive', 'negative', 'neutral']
    SENTIMENT_MAP = {sentiment: idx for idx, sentiment in enumerate(SENTIMENT_LABELS)}
    
    # Feature extraction settings
    SAVE_FORMAT = "pickle"  # or "numpy"
    VERBOSE = True

    # Model architectures
    # 'cahme'
    # 'm3fnet'
    MODEL_ARCHITECTURE = 'cahme'  # 'cahme'
    
    # M3F-Net specific configurations
    M3FNET_HIDDEN_DIM = 256
    M3FNET_SCALES = [1, 3, 5]  # Multi-scale convolution kernels
    M3FNET_ATTENTION_HEADS = 4
    M3FNET_LSTM_LAYERS = 2
    M3FNET_LSTM_DROPOUT = 0.2
    M3FNET_ATTENTION_DROPOUT = 0.1
    M3FNET_CLASSIFIER_DROPOUT_1 = 0.3
    M3FNET_CLASSIFIER_DROPOUT_2 = 0.2