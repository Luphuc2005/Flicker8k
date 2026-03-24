import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys
from pathlib import Path

# Thêm đường dẫn root để import các module
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.cnn_encoder import CNNEncoder
from models.transformer_encoder import TransformerEncoder
from models.transformer_decoder import TransformerDecoder
from models.caption_model import CaptionModel
from datasets.vocab import Vocabulary
from datasets.preprocessing import load_captions, filter_valid_images, flatten_data

def greedy_decode(model, image, vocab, max_len=20, device="cpu"):
    """
    Sinh caption bằng phương pháp Greedy Search.
    
    1. Lấy đặc trưng ảnh từ Encoder.
    2. Bắt đầu với token <start>.
    3. Dự đoán từ tiếp theo, chọn từ có xác suất cao nhất.
    4. Lặp lại cho đến khi gặp <end> hoặc đạt độ dài tối đa.
    """
    model.eval()
    with torch.no_grad():
        # 1. Lấy features từ ảnh
        features = model.encoder(image)
        features = model.transformer_encoder(features) # (1, 49, 512)

        # 2. Bắt đầu với <start>
        start_token = vocab.stoi["<start>"]
        captions = [start_token]
        
        for _ in range(max_len):
            # Chuyển list thành tensor
            caption_tensor = torch.LongTensor(captions).unsqueeze(0).to(device)
            
            # Predict
            output = model.decoder(caption_tensor, features)
            
            # Lấy từ cuối cùng được dự đoán (last token in sequence)
            last_word_logits = output[0, -1, :]
            predicted_id = last_word_logits.argmax().item()
            
            captions.append(predicted_id)
            
            # Nếu gặp <end> thì dừng
            if predicted_id == vocab.stoi["<end>"]:
                break
                
    # Chuyển ID thành chữ, bỏ <start> và <end>
    words = [vocab.itos[idx] for idx in captions if idx not in [vocab.stoi["<start>"], vocab.stoi["<end>"], vocab.stoi["<pad>"]]]
    return " ".join(words)

def load_model_and_vocab(checkpoint_path, data_dir, device):
    # 1. Load Vocab (Sử dụng hàm get_loaders để khớp 100% với lúc train)
    from datasets.data_loader import get_loaders
    print("Mô hình đang tải Vocabulary chuẩn...")
    
    # Chỉ lấy vocab, các tham số khác để mặc định
    _, vocab, _ = get_loaders(data_dir)
    
    # 3. Load Trọng số (Checkpoint)
    print(f"Đang tải trọng số từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 4. Tự động lấy cấu hình từ checkpoint nếu có
    checkpoint_config = checkpoint.get("config", {})
    max_len = checkpoint_config.get("max_len", 42) # Mặc định là 42 theo lỗi báo
    embed_dim = checkpoint_config.get("embed_dim", 512)
    num_heads = checkpoint_config.get("num_head", 2)
    ff_dim = checkpoint_config.get("ff_dim", 512)
    
    print(f"Cấu hình phát hiện: max_len={max_len}, embed_dim={embed_dim}")

    # 5. Khởi tạo Model với đúng thông số
    encoder = CNNEncoder(embed_dim=embed_dim)
    trans_enc = TransformerEncoder(embed_dim=embed_dim, num_heads=num_heads)
    decoder = TransformerDecoder(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_head=num_heads,
        ff_dim=ff_dim,
        max_len=max_len
    )
    model = CaptionModel(encoder, trans_enc, decoder).to(device)
    
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    return model, vocab

def main():
    # --- CẤU HÌNH ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Thay đổi các đường dẫn này cho đúng với máy của em
    CHECKPOINT = r"D:\HocTap\NCKH_ThayDoNhuTai\Thực nghiệm\Image_Captioning_Flickr8k\Checkpoints\epoch_9.pt" 
    DATA_DIR = r"D:\HocTap\NCKH_ThayDoNhuTai\Thực nghiệm\Image_Captioning_Flickr8k\data" 
    IMAGE_PATH = r"D:\HocTap\NCKH_ThayDoNhuTai\Thực nghiệm\Image_Captioning_Flickr8k\data\Images\1000268201_693b08cb0e.jpg" # Một ảnh bất kỳ để test
    
    # --- TIỀN XỬ LÝ ẢNH ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- RUN ---
    if not os.path.exists(CHECKPOINT):
        print(f"LỖI: Không tìm thấy file checkpoint tại {CHECKPOINT}")
        return

    model, vocab = load_model_and_vocab(CHECKPOINT, DATA_DIR, DEVICE)
    
    img = Image.open(IMAGE_PATH).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    caption = greedy_decode(model, img_tensor, vocab, device=DEVICE)
    
    print("\n" + "="*30)
    print(f"IMAGE: {IMAGE_PATH}")
    print(f"RESULT: {caption}")
    print("="*30)

if __name__ == "__main__":
    main()
