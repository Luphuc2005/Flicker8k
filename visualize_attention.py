import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import cv2

from config import config
from inference import load_model_and_vocab, greedy_decode

def plot_attention(image_path, words, attention_weights, save_path=None):
    """
    Vẽ ảnh gốc và bản đồ nhiệt Attention cho từng từ.
    words: list các từ đã dự đoán.
    attention_weights: tensor shape (seq_len, n_heads, 49)
    """
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    
    # Tính trung bình attention qua các heads
    # shape: (seq_len, 49)
    avg_attention = attention_weights.mean(dim=1)
    
    num_words = len(words)
    cols = 4
    rows = (num_words // cols) + (1 if num_words % cols != 0 else 0)
    
    fig = plt.figure(figsize=(15, 4 * rows))
    
    for i in range(num_words):
        plt.subplot(rows, cols, i + 1)
        plt.text(0, 1, words[i], color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(img_array)
        
        # Resize attention map (7x7) to match image size
        # avg_attention[i] has 49 elements
        attn_map = avg_attention[i].reshape(7, 7).numpy()
        attn_map = cv2.resize(attn_map, (img_array.shape[1], img_array.shape[0]))
        
        plt.imshow(attn_map, alpha=0.6, cmap='jet')
        plt.axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"XAI result saved to: {save_path}")
    plt.show()

def run_xai_example(image_path):
    device = config.DEVICE
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "epoch_9.pt")
    data_dir = config.DATA_DIR
    
    model, vocab = load_model_and_vocab(checkpoint_path, data_dir, device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    print(f"Bắt đầu phân tích XAI cho ảnh: {image_path}")
    caption, attention_weights = greedy_decode(model, img_tensor, vocab, device=device, return_attention=True)
    
    print(f"Dự đoán: {caption}")
    
    words = caption.split()
    save_name = f"xai_{os.path.basename(image_path).split('.')[0]}.png"
    save_path = os.path.join(config.EVAL_RESULTS_DIR, save_name)
    
    plot_attention(image_path, words, attention_weights, save_path=save_path)

if __name__ == "__main__":
    # Thay bằng path ảnh thực tế trong folder Images của em
    # Ví dụ chọn 1 ảnh ngẫu nhiên từ tập test hoặc chỉ định 1 ảnh
    import random
    from datasets.preprocessing import load_captions
    captions = load_captions(os.path.join(config.DATA_DIR, "captions.txt"))
    test_img = random.choice(list(captions.keys()))
    image_path = os.path.join(config.DATA_DIR, "Images", test_img)
    
    run_xai_example(image_path)
