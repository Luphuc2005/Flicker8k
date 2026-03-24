import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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

from config import config

def run_evaluation(checkpoint_path, data_dir):
    device = config.DEVICE
    
    # 1. Load Model & Vocab
    model, vocab = load_model_and_vocab(checkpoint_path, data_dir, device)
    model.eval()

    # 2. Lấy DataLoader tập Test chuẩn từ config
    print("Đang chuẩn bị dữ liệu tập Test (70/15/15 split)...")
    (train_loader, val_loader, test_loader), _, (n_train, n_val, n_test) = get_loaders(
        data_dir,
        split_dir=config.SPLIT_DIR,
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        seed=config.SEED,
        batch_size=config.BATCH_SIZE,
        vocab=vocab # Dùng vocab đã load từ checkpoint
    )
    
    print(f"Số lượng ảnh tập Test: {n_test} ảnh.")

    gts = {}
    res = {}
    
    # Để tính CIDEr/BLEU chuẩn COCO, ta cần tất cả Ground Truth của từng ảnh
    # Ta load lại captions_dict để lấy đủ 5 câu/ảnh
    all_captions = load_captions(os.path.join(data_dir, "captions.txt"))
    
    # Lấy danh sách image_paths từ test_dataset
    test_dataset = test_loader.dataset
    
    print(f"Đang chạy inference trên {len(test_dataset)} mẫu (captions)...")
    # Lưu ý: test_dataset có thể chứa nhiều entry cho cùng 1 ảnh nếu ta phẳng hóa.
    # Nhưng ở bước evaluation này, ta chỉ cần chạy 1 lần cho mỗi ảnh.
    
    processed_images = set()
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            img_path = test_dataset.image_paths[i]
            img_name = os.path.basename(img_path)
            
            if img_name in processed_images:
                continue
            
            processed_images.add(img_name)
            img_id = img_name
            
            # Ground Truth
            gts[img_id] = [{"caption": c} for c in all_captions[img_name]]
            
            # Prediction
            img_tensor, _ = test_dataset[i]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
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
        # (Meteor(), "METEOR"), 
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
    print(f"KẾT QUẢ ĐÁNH GIÁ (TRÊN TẬP TEST - {len(processed_images)} ẢNH):")
    for metric, val in final_scores.items():
        print(f"{metric:<10}: {val:.4f}")
    print("="*40)

if __name__ == "__main__":
    run_evaluation(os.path.join(config.CHECKPOINT_DIR, "epoch_9.pt"), config.DATA_DIR)
