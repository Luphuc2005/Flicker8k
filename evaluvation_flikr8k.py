import os
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from inference import load_model_and_vocab, greedy_decode
from datasets.data_loader import get_loaders
from datasets.preprocessing import load_captions

def run_evaluation(checkpoint_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model & Vocab
    # Lưu ý: model_and_vocab sẽ tự động load kiến trúc từ checkpoint
    model, vocab = load_model_and_vocab(checkpoint_path, data_dir, device)
    model.eval()

    # 2. Lấy danh sách ảnh tập Validation (Phải khớp SEED với lúc Train)
    # Chúng ta dùng chung logic chia từ get_loaders để đảm bảo tính thống nhất
    print("Đang chuẩn bị dữ liệu tập Test (Validation set)...")
    captions_file = os.path.join(data_dir, "captions.txt")
    image_dir = os.path.join(data_dir, "Images")
    
    # Load lại captions chuẩn
    all_captions = load_captions(captions_file)
    all_image_names = list(all_captions.keys())
    
    import random
    random.seed(42) # Khớp với seed mặc định lúc train
    random.shuffle(all_image_names)
    
    val_split = 0.2
    val_size = int(len(all_image_names) * val_split)
    test_images = all_image_names[:val_size] # Tập này model chưa học
    
    print(f"Tổng bộ dữ liệu: {len(all_image_names)} ảnh.")
    print(f"Số lượng ảnh tập Test: {len(test_images)} ảnh.")

    gts = {}
    res = {}
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Giới hạn số lượng ảnh test nếu muốn chạy nhanh (ví dụ test trên 200 ảnh đầu của tập Val)
    # test_images = test_images[:200] 

    print(f"Đang chạy inference trên {len(test_images)} ảnh...")
    with torch.no_grad():
        for img_name in tqdm(test_images):
            img_id = img_name
            
            # Ground Truth: Lấy tất cả captions của ảnh này
            gts[img_id] = [{"caption": c} for c in all_captions[img_name]]
            
            # Dự đoán từ Model (res)
            img_path = os.path.join(image_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            generated_cap = greedy_decode(model, img_tensor, vocab, device=device)
            res[img_id] = [{"caption": generated_cap}]

    # 3. Tokenize (Chuẩn COCO)
    print("Tokenizing...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    # 4. Tính toán các chỉ số
    print("Đang tính toán các chỉ số...")
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        # (Meteor(), "METEOR"), # Tạm tắt nếu Java lỗi
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    final_scores = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                final_scores[m] = sc
        else:
            final_scores[method] = score

    # 5. In kết quả
    print("\n" + "="*40)
    print(f"KẾT QUẢ ĐÁNH GIÁ (TRÊN TẬP TEST - {len(test_images)} ẢNH):")
    for metric, val in final_scores.items():
        print(f"{metric:<10}: {val:.4f}")
    print("="*40)

if __name__ == "__main__":
    CHECKPOINT = "Checkpoints/epoch_10.pt"
    DATA_DIR = "data"
    run_evaluation(CHECKPOINT, DATA_DIR)
