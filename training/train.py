import os
import csv
import torch
import wandb

# WandB service in notebook/Windows may need longer startup time.
os.environ.setdefault("WANDB__SERVICE_WAIT", "120")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, captions in loader:
        images = images.to(device)
        captions = captions.to(device)

        # shift caption
        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        outputs = model(images, inputs)

        loss = criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            targets.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    for images, captions in loader:
        images = images.to(device)
        captions = captions.to(device)

        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        outputs = model(images, inputs)

        loss = criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            targets.reshape(-1)
        )

        total_loss += loss.item()

    return total_loss / len(loader)


def _save_checkpoint(path, model, optimizer, scheduler, epoch, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "config": config,
        },
        path,
    )


def _load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    return start_epoch


def _log_sample_images(run, loader, step, max_images=4):
    if run is None or loader is None:
        return
    try:
        images, _ = next(iter(loader))
        n = min(max_images, images.size(0))
        preview = []
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        for i in range(n):
            # Convert CHW tensor to HWC for WandB image logging.
            img = images[i].detach().cpu()
            img = img * std + mean  # de-normalize to viewable RGB range
            img = img.permute(1, 2, 0)
            img = torch.clamp(img, 0.0, 1.0).numpy()
            preview.append(wandb.Image(img, caption=f"sample_{i}"))
        wandb.log({"train_samples": preview}, step=step)
    except Exception as e:
        print(f"[wandb] Skip image logging: {e}")


def _upload_checkpoint_artifact(run, ckpt_path, epoch):
    if run is None:
        return
    try:
        artifact = wandb.Artifact(name=f"checkpoint-epoch-{epoch+1}", type="model")
        artifact.add_file(ckpt_path)
        run.log_artifact(artifact, aliases=["latest", f"epoch_{epoch+1}"])
    except Exception as e:
        print(f"[wandb] Skip checkpoint artifact upload: {e}")


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    device,
    epochs,
    config=None,
    checkpoint_dir="checkpoints",
    resume_path=None,
    use_wandb=True,
    wandb_mode="offline",
    wandb_project="image-captioning",
    wandb_name="",
    wandb_notes="",
    log_images=True,
    upload_checkpoints_to_wandb=True,
    metrics_csv_path=None,
):
    start_epoch = 0
    if resume_path:
        start_epoch = _load_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )

    if start_epoch >= epochs:
        raise ValueError(
            f"Khong co epoch nao de train: start_epoch={start_epoch}, epochs={epochs}. "
            "Tang --epochs hoac dat --resume none de train lai tu dau."
        )

    run = None
    if use_wandb:
        try:
            run = wandb.init(
                project=wandb_project,
                config=config,
                mode=wandb_mode,
                name=wandb_name or None,
                notes=wandb_notes or None,
                settings=wandb.Settings(init_timeout=120),
                reinit=True,
            )
            wandb.watch(model, log="all", log_freq=100)
        except Exception as e:
            print(f"[wandb] Init failed in {wandb_mode} mode ({e}).")
            if wandb_mode == "online":
                try:
                    print("[wandb] Retry with offline mode.")
                    run = wandb.init(
                        project=wandb_project,
                        config=config,
                        mode="offline",
                        name=wandb_name or None,
                        notes=wandb_notes or None,
                        settings=wandb.Settings(init_timeout=120),
                        reinit=True,
                    )
                    wandb.watch(model, log="all", log_freq=100)
                except Exception as e2:
                    print(f"[wandb] Offline init also failed ({e2}). Continue without wandb.")
                    run = None
            else:
                print("[wandb] Continue without wandb.")
                run = None

    history = []
    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss = validate_one_epoch(
            model, val_loader, criterion, device
        ) if val_loader else 0

        if scheduler:
            scheduler.step()

        if run is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            if log_images:
                source_loader = val_loader if val_loader else train_loader
                _log_sample_images(run, source_loader, step=epoch)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss = {train_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}"
        )

        metrics = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(metrics)

        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
        _save_checkpoint(
            ckpt_path, model, optimizer, scheduler, epoch, config
        )
        print(f"[checkpoint] Saved: {ckpt_path}")

        if upload_checkpoints_to_wandb and run is not None:
            _upload_checkpoint_artifact(run, ckpt_path, epoch)

        if metrics_csv_path:
            os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
            write_header = not os.path.exists(metrics_csv_path)
            with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["epoch", "train_loss", "val_loss", "lr"]
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(metrics)

    if run is not None:
        wandb.finish()

    return history