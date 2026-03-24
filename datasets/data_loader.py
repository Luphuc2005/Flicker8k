import os
import random
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .preprocessing import load_captions, filter_valid_images
from .dataset import FlickrDataset
from .collate import collate_fn
from .vocab import Vocabulary

def get_loaders(
    data_dir,
    image_dir=None,
    captions_file=None,
    vocab=None,
    batch_size=32,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    seed=42,
    freq_threshold=5,
    num_workers=2,
    transform=None
):
    """
    Hàm chuẩn để lấy DataLoaders với 3 tập: Train, Val, Test.
    Chia dữ liệu theo Image ID để tránh rò rỉ thông tin.
    """
    if image_dir is None:
        image_dir = os.path.join(data_dir, "Images")
    if captions_file is None:
        captions_file = os.path.join(data_dir, "captions.txt")

    # 1. Load data
    captions_dict = load_captions(captions_file)

    # 2. Chia theo Image ID
    all_image_names = sorted(list(captions_dict.keys())) # Sắp xếp để đảm bảo thứ tự ban đầu cố định
    
    split_file = os.path.join(data_dir, f"splits_seed_{seed}.json")
    if os.path.exists(split_file):
        print(f"Loading existing splits from {split_file}")
        with open(split_file, "r") as f:
            splits = json.load(f)
        train_names = set(splits["train"])
        val_names = set(splits["val"])
        test_names = set(splits["test"])
    else:
        print(f"Creating new splits with seed {seed}...")
        random.seed(seed)
        random.shuffle(all_image_names)

        num_images = len(all_image_names)
        train_end = int(num_images * train_split)
        val_end = train_end + int(num_images * val_split)

        train_names_list = all_image_names[:train_end]
        val_names_list = all_image_names[train_end:val_end]
        test_names_list = all_image_names[val_end:]
        
        # Lưu lại để lần sau dùng đúng tập này
        with open(split_file, "w") as f:
            json.dump({
                "train": train_names_list,
                "val": val_names_list,
                "test": test_names_list
            }, f)
            
        train_names = set(train_names_list)
        val_names = set(val_names_list)
        test_names = set(test_names_list)

    # 3. Phẳng hóa dữ liệu
    train_img_paths, train_caps = [], []
    val_img_paths, val_caps = [], []
    test_img_paths, test_caps = [], []

    for img_name, caps in captions_dict.items():
        path = os.path.join(image_dir, img_name)
        for c in caps:
            if img_name in train_names:
                train_img_paths.append(path)
                train_caps.append(c)
            elif img_name in val_names:
                val_img_paths.append(path)
                val_caps.append(c)
            else:
                test_img_paths.append(path)
                test_caps.append(c)

    # 4. Build Vocab
    if vocab is None:
        vocab = Vocabulary(freq_threshold=freq_threshold)
        vocab.build_vocab(train_caps)

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
    test_dataset = FlickrDataset(test_img_paths, test_caps, vocab, transform=transform)

    # 7. Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, collate_fn=collate_fn)

    return (train_loader, val_loader, test_loader), vocab, \
           (len(train_names), len(val_names), len(test_names))
