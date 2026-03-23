import argparse
from datetime import datetime
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.collate import collate_fn
from datasets.dataset import FlickrDataset
from datasets.preprocessing import filter_valid_images, flatten_data, load_captions
from datasets.vocab import Vocabulary
from models.caption_model import CaptionModel
from models.cnn_encoder import CNNEncoder
from models.transformer_decoder import TransformerDecoder
from models.transformer_encoder import TransformerEncoder
from training.loss import get_loss
from training.scheduler import get_scheduler
from training.train import train_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_data_paths(data_dir: str, image_dir: str, captions_file: str):
    # Honor explicit paths only when both actually exist.
    if image_dir and captions_file:
        if Path(image_dir).is_dir() and Path(captions_file).is_file():
            return image_dir, captions_file

        print("[warn] image_dir/captions_file provided but not found, fallback to auto-detect...")

    image_dir_names = {
        "images",
        "flickr8k_dataset",
        "flicker8k_dataset",
        "flickr8k-images",
    }
    caption_patterns = [
        "captions.txt",
        "*caption*.txt",
        "flickr8k.token.txt",
        "*token*.txt",
    ]

    candidates = []
    if data_dir:
        candidates.append(Path(data_dir))
    candidates.extend(
        [
            REPO_ROOT / "data",
            Path("/kaggle/input"),
            Path("/kaggle/working/data"),
        ]
    )

    for base in candidates:
        if not base.exists():
            continue

        if (base / "Images").exists() and (base / "captions.txt").exists():
            return str(base / "Images"), str(base / "captions.txt")

        txts = []
        for pattern in caption_patterns:
            txts.extend(base.rglob(pattern))

        # Keep unique files and prioritize exact captions.txt first.
        txts = sorted({p for p in txts if p.is_file()}, key=lambda p: (p.name.lower() != "captions.txt", len(str(p))))

        imgs = [p for p in base.rglob("*") if p.is_dir() and p.name.lower() in image_dir_names]
        if txts and imgs:
            return str(imgs[0]), str(txts[0])

    raise FileNotFoundError(
        "Khong tim thay data/captions. Hay truyen --data-dir hoac --image-dir va --captions-file dung duong dan."
    )


def get_resume_path(resume_mode: str, checkpoint_dir: str):
    if resume_mode == "none":
        return None
    if resume_mode != "latest":
        return resume_mode

    ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt")))
    return ckpts[-1] if ckpts else None


def _cfg_get(cfg: dict, key: str, default):
    return cfg.get(key, cfg.get(key.replace("_", "-"), default))


def _build_parser(config_defaults: dict):
    parser = argparse.ArgumentParser(description="Run full Flickr8k training pipeline for Kaggle")
    parser.add_argument("--config", type=str, default="", help="Duong dan file JSON chua cau hinh")
    parser.add_argument("--data-dir", type=str, default=_cfg_get(config_defaults, "data_dir", ""), help="Folder chua Images va captions.txt")
    parser.add_argument("--image-dir", type=str, default=_cfg_get(config_defaults, "image_dir", ""), help="Duong dan thu muc Images")
    parser.add_argument("--captions-file", type=str, default=_cfg_get(config_defaults, "captions_file", ""), help="Duong dan captions.txt")
    parser.add_argument("--output-dir", type=str, default=_cfg_get(config_defaults, "output_dir", "/kaggle/working/outputs"))
    parser.add_argument("--config-out-dir", type=str, default=_cfg_get(config_defaults, "config_out_dir", ""), help="Thu muc luu config da resolve cho moi run")
    parser.add_argument("--epochs", type=int, default=_cfg_get(config_defaults, "epochs", 10))
    parser.add_argument("--batch-size", type=int, default=_cfg_get(config_defaults, "batch_size", 32))
    parser.add_argument("--lr", type=float, default=_cfg_get(config_defaults, "lr", 1e-4))
    parser.add_argument("--embed-dim", type=int, default=_cfg_get(config_defaults, "embed_dim", 512))
    parser.add_argument("--num-head", type=int, default=_cfg_get(config_defaults, "num_head", 2))
    parser.add_argument("--ff-dim", type=int, default=_cfg_get(config_defaults, "ff_dim", 512))
    parser.add_argument("--freq-threshold", type=int, default=_cfg_get(config_defaults, "freq_threshold", 5))
    parser.add_argument("--val-split", type=float, default=_cfg_get(config_defaults, "val_split", 0.2))
    parser.add_argument("--num-workers", type=int, default=_cfg_get(config_defaults, "num_workers", 2))
    parser.add_argument("--seed", type=int, default=_cfg_get(config_defaults, "seed", 42))
    parser.add_argument("--resume", type=str, default=_cfg_get(config_defaults, "resume", "latest"), help="latest | none | /path/to/ckpt.pt")
    parser.add_argument("--wandb-mode", type=str, default=_cfg_get(config_defaults, "wandb_mode", "offline"), choices=["offline", "online", "disabled"])
    parser.add_argument("--wandb-project", type=str, default=_cfg_get(config_defaults, "wandb_project", "image-captioning"))
    parser.add_argument("--wandb-name", type=str, default=_cfg_get(config_defaults, "wandb_name", ""), help="Ten run hien thi tren WandB")
    parser.add_argument("--wandb-notes", type=str, default=_cfg_get(config_defaults, "wandb_notes", ""), help="Ghi chu cho run tren WandB")
    parser.add_argument("--upload-checkpoints", dest="upload_checkpoints", action="store_true", help="Upload checkpoint moi epoch len WandB Artifacts")
    parser.add_argument("--no-upload-checkpoints", dest="upload_checkpoints", action="store_false", help="Khong upload checkpoint len WandB")
    parser.add_argument("--log-images", dest="log_images", action="store_true", help="Bat log sample images len WandB")
    parser.add_argument("--no-log-images", dest="log_images", action="store_false", help="Tat log sample images len WandB")
    parser.set_defaults(log_images=bool(_cfg_get(config_defaults, "log_images", False)))
    parser.set_defaults(upload_checkpoints=bool(_cfg_get(config_defaults, "upload_checkpoints", True)))
    return parser.parse_args()


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="")
    pre_args, _ = pre_parser.parse_known_args()

    config_defaults = {}
    if pre_args.config:
        with open(pre_args.config, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("Noi dung file --config phai la JSON object.")
            config_defaults = loaded

    return _build_parser(config_defaults)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    config_out_dir = Path(args.config_out_dir) if args.config_out_dir else (output_dir / "configs")
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_out_dir.mkdir(parents=True, exist_ok=True)

    image_dir, captions_file = resolve_data_paths(args.data_dir, args.image_dir, args.captions_file)

    captions_dict = load_captions(captions_file)
    captions_dict = filter_valid_images(image_dir, captions_dict)
    image_paths, captions = flatten_data(image_dir, captions_dict)

    vocab = Vocabulary(freq_threshold=args.freq_threshold)
    vocab.build_vocab(captions)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = FlickrDataset(image_paths, captions, vocab, transform=transform)
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len = max(len(vocab.numericalize(c)) + 2 for c in captions)

    encoder = CNNEncoder(embed_dim=args.embed_dim)
    trans_enc = TransformerEncoder(embed_dim=args.embed_dim, num_heads=args.num_head)
    decoder = TransformerDecoder(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        num_head=args.num_head,
        ff_dim=args.ff_dim,
        max_len=max_len,
    )
    model = CaptionModel(encoder, trans_enc, decoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = get_loss()
    scheduler = get_scheduler(optimizer)

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "embed_dim": args.embed_dim,
        "num_head": args.num_head,
        "ff_dim": args.ff_dim,
        "freq_threshold": args.freq_threshold,
        "val_split": args.val_split,
        "seed": args.seed,
        "max_len": int(max_len),
        "dataset_size": len(dataset),
        "train_size": train_size,
        "val_size": val_size,
        "device": str(device),
        "image_dir": image_dir,
        "captions_file": captions_file,
        "wandb_mode": args.wandb_mode,
        "wandb_project": args.wandb_project,
        "wandb_name": args.wandb_name,
        "wandb_notes": args.wandb_notes,
        "upload_checkpoints": args.upload_checkpoints,
        "log_images": args.log_images,
    }

    config_path = output_dir / "train_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    resolved_config_path = config_out_dir / f"run_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(resolved_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    resume_path = get_resume_path(args.resume, str(checkpoint_dir))
    use_wandb = args.wandb_mode != "disabled"
    wandb_mode = "offline" if args.wandb_mode == "disabled" else args.wandb_mode

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        config=config,
        checkpoint_dir=str(checkpoint_dir),
        resume_path=resume_path,
        use_wandb=use_wandb,
        wandb_mode=wandb_mode,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_notes=args.wandb_notes,
        log_images=args.log_images,
        upload_checkpoints_to_wandb=args.upload_checkpoints,
        metrics_csv_path=str(output_dir / "metrics.csv"),
    )

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("Done. Artifacts saved to:")
    print(f"- {output_dir}")
    print(f"- {config_path}")
    print(f"- {output_dir / 'metrics.csv'}")
    print(f"- {output_dir / 'history.json'}")
    print(f"- {checkpoint_dir}")
    print(f"- {resolved_config_path}")


if __name__ == "__main__":
    main()
