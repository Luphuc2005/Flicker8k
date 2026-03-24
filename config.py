import torch
import os

class Config:
    # --- BATHS ---
    DATA_DIR = "data"
    IMAGE_DIR = os.path.join(DATA_DIR, "Images")
    CAPTIONS_FILE = os.path.join(DATA_DIR, "captions.txt")
    CHECKPOINT_DIR = "Checkpoints"
    
    # --- DATA SPLIT ---
    # Tập Test sẽ được lưu ở đây để tránh lỗi Read-only trên Kaggle
    SPLIT_DIR = "." # Mặc định lưu ở thư mục gốc của repo
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    SEED = 42
    
    # --- DATALOADER ---
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    FREQ_THRESHOLD = 5
    
    # --- MODEL ARCHITECTURE ---
    EMBED_DIM = 512
    NUM_HEADS = 8
    FF_DIM = 2048
    MAX_LEN = 42  # Độ dài tối đa của caption (bao gồm <start> và <end>)
    
    # --- TRAINING ---
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
