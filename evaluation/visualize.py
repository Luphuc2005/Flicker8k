import os
import torch
import random
from PIL import Image
from torchvision import transforms

# Đảm bảo import được các module từ thư mục gốc
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import load_model_and_vocab, greedy_decode, beam_search_decode
from datasets.data_loader import get_loaders
from datasets.preprocessing import load_captions

def run_visualization(checkpoint_path, data_dir, num_samples=5, beam_size=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model & Vocab
    model, vocab = load_model_and_vocab(checkpoint_path, data_dir, device)
    model.eval()

    # 2. Lấy danh sách ảnh tập Validation (khớp seed=42 để đúng chuẩn Test set)
    print("Đang quét tập dữ liệu Test...")
    captions_file = os.path.join(data_dir, "captions.txt")
    image_dir = os.path.join(data_dir, "Images")
    
    all_captions = load_captions(captions_file)
    all_image_names = list(all_captions.keys())
    
    # Chia y hệt như lúc Train/Eval
    random.seed(42)
    random.shuffle(all_image_names)
    
    val_split = 0.2
    val_size = int(len(all_image_names) * val_split)
    test_images = all_image_names[:val_size]
    
    # Chọn ngẫu nhiên vài ảnh từ tập Test để "khám nghiệm"
    samples = random.sample(test_images, num_samples)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"\n" + "="*60)
    print(f"BÁO CÁO ĐÁNH GIÁ ĐỊNH TÍNH (QUALITATIVE ANALYSIS)")
    print(f"Số lượng mẫu thử: {num_samples} ảnh tập Test")
    print("="*60 + "\n")

    for i, img_name in enumerate(samples):
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # 1. Greedy Search
        pred_greedy = greedy_decode(model, img_tensor, vocab, device=device)
        
        # 2. Beam Search
        pred_beam = beam_search_decode(model, img_tensor, vocab, beam_size=beam_size, device=device)
        
        # 3. Ground Truths (Lấy câu đầu tiên làm mẫu)
        gts = all_captions[img_name]
        
        print(f"[{i+1}] ẢNH: {img_name}")
        print(f"  > [Ground Truth]: {gts[0]}")
        print(f"  > [Greedy Gen]  : {pred_greedy}")
        print(f"  > [Beam Gen k=5]: {pred_beam}")
        print("-" * 60)

if __name__ == "__main__":
    # Cấu hình đường dẫn
    CHECKPOINT = "Checkpoints/epoch_9.pt"
    DATA_DIR = "data"
    
    if not os.path.exists(CHECKPOINT):
        print(f"Lỗi: Không tìm thấy checkpoint tại {CHECKPOINT}")
    else:
        run_visualization(CHECKPOINT, DATA_DIR, num_samples=5, beam_size=5)
