import os
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .preprocessing import load_captions, filter_valid_images
from .dataset import FlickrDataset
from .collate import collate_fn
from .vocab import Vocabulary

def get_loaders(
    data_dir,
    vocab=None,
    batch_size=32,
    val_split=0.2,
    seed=42,
    freq_threshold=5,
    num_workers=2,
    transform=None
):
    """
    Hàm chuẩn để lấy DataLoaders mà không bị rò rỉ dữ liệu (Data Leakage).
    Chia dữ liệu theo Image ID thay vì chia theo từng caption.
    """
    image_dir = os.path.join(data_dir, "Images")
    captions_file = os.path.join(data_dir, "captions.txt")

    # 1. Load data
    captions_dict = load_captions(captions_file)
    # captions_dict = filter_valid_images(image_dir, captions_dict) # Bật nếu muốn lọc ảnh lỗi

    # 2. Chia theo Image ID (Fix Image-level leakage)
    all_image_names = list(captions_dict.keys())
    random.seed(seed)
    random.shuffle(all_image_names)

    val_size = int(len(all_image_names) * val_split)
    val_names = set(all_image_names[:val_size])
    train_names = set(all_image_names[val_size:])

    # 3. Phẳng hóa dữ liệu cho từng tập
    train_img_paths, train_caps = [], []
    val_img_paths, val_caps = [], []

    for img_name, caps in captions_dict.items():
        for c in caps:
            # Lưu ý: c lúc này đã có thể chứa <start> <end> tùy vào preprocessing.py
            # Thầy khuyên nên để nguyên bản và xử lý token trong Dataset class.
            path = os.path.join(image_dir, img_name)
            if img_name in train_names:
                train_img_paths.append(path)
                train_caps.append(c)
            else:
                val_img_paths.append(path)
                val_caps.append(c)

    # 4. Tạo hoặc Build Vocab
    if vocab is None:
        vocab = Vocabulary(freq_threshold=freq_threshold)
        vocab.build_vocab(train_caps) # Chỉ build vocab dựa trên tập Train!

    # 5. Transforms
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # 6. Datasets
    train_dataset = FlickrDataset(train_img_paths, train_caps, vocab, transform=transform)
    val_dataset = FlickrDataset(val_img_paths, val_caps, vocab, transform=transform)

    # 7. Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, vocab, len(all_image_names), len(train_names), len(val_names), list(train_caps) + list(val_caps)
