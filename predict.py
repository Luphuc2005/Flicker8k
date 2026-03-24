import torch
import os
import argparse
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from inference import load_model_and_vocab, greedy_decode

from config import config

def predict_single_image(image_path, checkpoint_path=None, data_dir=None):
    device = config.DEVICE
    checkpoint_path = checkpoint_path or os.path.join(config.CHECKPOINT_DIR, "epoch_9.pt")
    data_dir = data_dir or config.DATA_DIR
    
    # 1. Load Model & Vocab
    model, vocab = load_model_and_vocab(checkpoint_path, data_dir, device)
    model.eval()

    # 2. Tiền xử lý ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not os.path.exists(image_path):
        print(f"LỖI: Không tìm thấy ảnh tại {image_path}")
        return

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 3. Sinh Caption
    print("Mô hình đang suy luận...")
    caption = greedy_decode(model, img_tensor, vocab, device=device)

    # 4. Hiển thị kết quả
    print("\n" + "="*40)
    print(f"ẢNH: {os.path.basename(image_path)}")
    print(f"KẾT QUẢ: {caption}")
    print("="*40)

    # (Tùy chọn) Hiển thị ảnh nếu có môi trường đồ họa
    # plt.imshow(img)
    # plt.title(f"Predicted: {caption}")
    # plt.axis('off')
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dự đoán caption cho 1 ảnh đơn lẻ.")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn tới file ảnh.")
    parser.add_argument("--checkpoint", type=str, default="Checkpoints/epoch_9.pt", help="Đường dẫn tới checkpoint model.")
    parser.add_argument("--data_dir", type=str, default="data", help="Thư mục chứa dữ liệu (để load vocab).")
    
    args = parser.parse_args()

    predict_single_image(args.image, args.checkpoint, args.data_dir)
