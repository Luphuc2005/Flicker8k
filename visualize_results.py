import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import random
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import config
from inference import load_model_and_vocab, greedy_decode
from datasets.data_loader import get_loaders
from datasets.preprocessing import load_captions

def visualize_test_results(num_samples=5, save_to_dir=None):
    device = config.DEVICE
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "epoch_9.pt")
    data_dir = config.DATA_DIR
    
    # 1. Load Model & Vocab
    model, vocab = load_model_and_vocab(checkpoint_path, data_dir, device)
    model.eval()

    # 2. Lấy Test Set chuẩn
    (train_loader, val_loader, test_loader), _, _ = get_loaders(
        data_dir,
        split_dir=config.SPLIT_DIR,
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        seed=config.SEED,
        vocab=vocab
    )
    
    test_dataset = test_loader.dataset
    all_captions = load_captions(os.path.join(data_dir, "captions.txt"))
    
    # 3. Chọn ngẫu nhiên các ảnh duy nhất từ tập test
    # Lưu ý: dataset phẳng hóa nên 1 ảnh có thể xuất hiện nhiều lần
    unique_img_paths = list(set(test_dataset.image_paths))
    samples = random.sample(unique_img_paths, min(num_samples, len(unique_img_paths)))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if save_to_dir is None:
        save_to_dir = config.EVAL_RESULTS_DIR
    os.makedirs(save_to_dir, exist_ok=True)

    print(f"Đang visualize {len(samples)} ảnh từ tập Test...")
    
    plt.figure(figsize=(20, 5 * len(samples)))
    
    for i, img_path in enumerate(samples):
        img_name = os.path.basename(img_path)
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        # Dự đoán
        with torch.no_grad():
            pred_caption = greedy_decode(model, img_tensor, vocab, device=device)

        # Ground Truth
        gt_captions = all_captions.get(img_name, ["No GT found"])
        
        # Plot
        plt.subplot(len(samples), 1, i + 1)
        plt.imshow(img_pil)
        title = f"GT: {gt_captions[0]}\nPRED: {pred_caption}"
        plt.title(title, fontsize=12, loc='left')
        plt.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_to_dir, f"test_visualization_{random.randint(100,999)}.png")
    plt.savefig(save_path)
    print(f"Đã lưu kết quả visualize tại: {save_path}")
    # plt.show() # Tạm tắt nếu chạy trong môi trường không UI

if __name__ == "__main__":
    visualize_test_results(num_samples=5)
