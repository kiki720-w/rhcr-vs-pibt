import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MAPFDataset
from modles import MAPF_ResUNet


@dataclass
class TrainConfig:
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    NUM_EPOCHS: int = 30
    NUM_WORKERS: int = 0
    GRAD_CLIP: float = 1.0

    LABEL_MODE: str = "downsample_stay"
    STAY_KEEP_PROB: float = 0.2

    HEATMAP_LOSS_WEIGHT: float = 0.5
    EARLY_STOP_PATIENCE: int = 5

    USE_AMP: bool = True
    SEED: int = 42

    TRAIN_DIR: str = "./dataset_v2_random/train"
    VAL_DIR: str = "./dataset_v2_random/val"
    SAVE_DIR: str = "./checkpoints_multi"

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_valid_accuracy(logits, labels, ignore_index=-1):
    preds = torch.argmax(logits, dim=1)
    valid_mask = labels != ignore_index

    valid_count = valid_mask.sum().item()
    if valid_count == 0:
        return 0.0, 0

    correct = (preds[valid_mask] == labels[valid_mask]).sum().item()
    return correct / valid_count, valid_count


def train_one_epoch(model, loader, optimizer, criterion_action, criterion_heatmap, device, scaler, cfg):
    model.train()

    total_loss = 0.0
    total_action_loss = 0.0
    total_heatmap_loss = 0.0
    total_acc_sum = 0.0
    total_valid_pixels = 0

    pbar = tqdm(loader, desc="[Train]", leave=False)

    for step, batch in enumerate(pbar):
        map_feat = batch["map_feat"].to(device, non_blocking=True)
        agent_feat = batch["agent_feat"].to(device, non_blocking=True)
        res_feat = batch["res_feat"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        heatmap_target = batch["heatmap_target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        use_amp = cfg.USE_AMP and device.type == "cuda"

        with torch.cuda.amp.autocast(enabled=use_amp):
            action_logits, pred_heatmap = model(
                map_feat,
                agent_feat,
                res_feat,
                return_aux=True,
            )

            loss_action = criterion_action(action_logits, labels)
            loss_heatmap = criterion_heatmap(pred_heatmap, heatmap_target)

            loss = loss_action + cfg.HEATMAP_LOSS_WEIGHT * loss_heatmap

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

        batch_acc, valid_pixels = compute_valid_accuracy(action_logits.detach(), labels)

        total_loss += loss.item()
        total_action_loss += loss_action.item()
        total_heatmap_loss += loss_heatmap.item()
        total_acc_sum += batch_acc * valid_pixels
        total_valid_pixels += valid_pixels

        avg_loss = total_loss / (step + 1)
        avg_action_loss = total_action_loss / (step + 1)
        avg_heatmap_loss = total_heatmap_loss / (step + 1)
        avg_acc = total_acc_sum / max(1, total_valid_pixels)

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "act": f"{avg_action_loss:.4f}",
            "hm": f"{avg_heatmap_loss:.4f}",
            "acc": f"{avg_acc:.4f}",
        })

    return (
        total_loss / len(loader),
        total_action_loss / len(loader),
        total_heatmap_loss / len(loader),
        total_acc_sum / max(1, total_valid_pixels),
    )


@torch.no_grad()
def validate_one_epoch(model, loader, criterion_action, criterion_heatmap, device, cfg):
    model.eval()

    total_loss = 0.0
    total_action_loss = 0.0
    total_heatmap_loss = 0.0
    total_acc_sum = 0.0
    total_valid_pixels = 0

    for batch in tqdm(loader, desc="[Val]", leave=False):
        map_feat = batch["map_feat"].to(device, non_blocking=True)
        agent_feat = batch["agent_feat"].to(device, non_blocking=True)
        res_feat = batch["res_feat"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        heatmap_target = batch["heatmap_target"].to(device, non_blocking=True)

        use_amp = cfg.USE_AMP and device.type == "cuda"

        with torch.cuda.amp.autocast(enabled=use_amp):
            action_logits, pred_heatmap = model(
                map_feat,
                agent_feat,
                res_feat,
                return_aux=True,
            )

            loss_action = criterion_action(action_logits, labels)
            loss_heatmap = criterion_heatmap(pred_heatmap, heatmap_target)

            loss = loss_action + cfg.HEATMAP_LOSS_WEIGHT * loss_heatmap

        batch_acc, valid_pixels = compute_valid_accuracy(action_logits, labels)

        total_loss += loss.item()
        total_action_loss += loss_action.item()
        total_heatmap_loss += loss_heatmap.item()
        total_acc_sum += batch_acc * valid_pixels
        total_valid_pixels += valid_pixels

    return (
        total_loss / len(loader),
        total_action_loss / len(loader),
        total_heatmap_loss / len(loader),
        total_acc_sum / max(1, total_valid_pixels),
    )


def train():
    cfg = TrainConfig()
    set_seed(cfg.SEED)

    device = torch.device(cfg.DEVICE)
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    print(f"🔥 使用设备: {device}")
    print(f"📁 Train dir: {cfg.TRAIN_DIR}")
    print(f"📁 Val dir:   {cfg.VAL_DIR}")
    print(f"⚖️ Heatmap loss weight: {cfg.HEATMAP_LOSS_WEIGHT}")
    print(f"🛑 Early stopping patience: {cfg.EARLY_STOP_PATIENCE}")

    train_dataset = MAPFDataset(
        data_dir=cfg.TRAIN_DIR,
        return_raw=False,
        strict=True,
        label_mode=cfg.LABEL_MODE,
        stay_keep_prob=cfg.STAY_KEEP_PROB,
        clamp_heatmap=True,
    )

    val_dataset = MAPFDataset(
        data_dir=cfg.VAL_DIR,
        return_raw=False,
        strict=True,
        label_mode=cfg.LABEL_MODE,
        stay_keep_prob=cfg.STAY_KEEP_PROB,
        clamp_heatmap=True,
    )

    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Val samples:   {len(val_dataset)}")

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    model = MAPF_ResUNet(
        num_actions=5,
        use_aux_head=True,
        dropout_p=0.10,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    criterion_action = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_heatmap = nn.BCEWithLogitsLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.USE_AMP and device.type == "cuda"))

    best_val_loss = float("inf")
    best_val_acc = 0.0
    no_improve_count = 0

    for epoch in range(cfg.NUM_EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{cfg.NUM_EPOCHS} =====")

        train_loss, train_action_loss, train_heatmap_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion_action,
            criterion_heatmap,
            device,
            scaler,
            cfg,
        )

        val_loss, val_action_loss, val_heatmap_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion_action,
            criterion_heatmap,
            device,
            cfg,
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"📈 Epoch {epoch + 1} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} "
            f"(Action {train_action_loss:.4f}, Heatmap {train_heatmap_loss:.4f}) | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} "
            f"(Action {val_action_loss:.4f}, Heatmap {val_heatmap_loss:.4f}) | "
            f"Val Acc: {val_acc:.4f}"
        )

        latest_path = os.path.join(cfg.SAVE_DIR, "latest_model_multi.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_action_loss": val_action_loss,
            "val_heatmap_loss": val_heatmap_loss,
            "val_acc": val_acc,
            "config": cfg.__dict__,
        }, latest_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = max(best_val_acc, val_acc)
            no_improve_count = 0

            best_path = os.path.join(cfg.SAVE_DIR, "best_model_multi.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_action_loss": val_action_loss,
                "val_heatmap_loss": val_heatmap_loss,
                "val_acc": val_acc,
                "config": cfg.__dict__,
            }, best_path)

            print(f"💾 最佳多任务模型已保存: {best_path}")
        else:
            best_val_acc = max(best_val_acc, val_acc)
            no_improve_count += 1
            print(f"⏳ Val Loss 没提升: {no_improve_count}/{cfg.EARLY_STOP_PATIENCE}")

            if no_improve_count >= cfg.EARLY_STOP_PATIENCE:
                print("🛑 Early stopping triggered.")
                break

    print("\n🎉 多任务训练完成！")
    print(f"🏆 Best Val Loss: {best_val_loss:.4f}")
    print(f"🏆 Best Val Acc:  {best_val_acc:.4f}")


if __name__ == "__main__":
    train()