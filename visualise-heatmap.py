import os
import random

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from dataset import MAPFDataset
from modles import MAPF_ResUNet


# =========================
# 1. 配置
# =========================
DATA_DIR = "./dataset_v2_random/val"
MODEL_PATH = "./checkpoints_multi/best_model_multi.pth"
SAVE_DIR = "./visual_outputs_rhcr"

NUM_SAMPLES = 5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. 加载模型
# =========================
def load_model():
    model = MAPF_ResUNet(
        num_actions=5,
        use_aux_head=True,
        dropout_p=0.10,
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Checkpoint val loss: {checkpoint.get('val_loss', 'unknown')}")
        print(f"Checkpoint val acc: {checkpoint.get('val_acc', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


# =========================
# 3. 可视化一个样本
# =========================
def visualize_one_sample(sample, model, save_path, sample_id=0):
    map_feat = sample["map_feat"].unsqueeze(0).to(DEVICE)
    agent_feat = sample["agent_feat"].unsqueeze(0).to(DEVICE)
    res_feat = sample["res_feat"].unsqueeze(0).to(DEVICE)

    label = sample["label"]  # [H, W]
    gt_heatmap = sample["heatmap_target"].squeeze(0)  # [H, W]

    with torch.no_grad():
        action_logits, heatmap_logits = model(
            map_feat,
            agent_feat,
            res_feat,
            return_aux=True,
        )

        action_pred = torch.argmax(action_logits, dim=1).squeeze(0).cpu()
        pred_heatmap = torch.sigmoid(heatmap_logits).squeeze(0).squeeze(0).cpu()

    obstacle_map = sample["map_feat"][0].cpu()
    agent_pos = sample["agent_feat"][0].cpu()
    goal_pos = sample["agent_feat"][1].cpu()

    valid_mask = label != -1

    # 只在有效 agent 位置显示动作
    gt_action_show = torch.full_like(label, -1)
    pred_action_show = torch.full_like(label, -1)

    gt_action_show[valid_mask] = label[valid_mask]
    pred_action_show[valid_mask] = action_pred[valid_mask]

    error_heatmap = torch.abs(pred_heatmap - gt_heatmap)

    # heatmap threshold 展示版，方便报告更清楚
    pred_heatmap_clean = pred_heatmap.clone()
    pred_heatmap_clean[pred_heatmap_clean < 0.15] = 0.0

    # =========================
    # 画图
    # =========================
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # 1. obstacle map
    axes[0, 0].imshow(obstacle_map, cmap="gray_r")
    axes[0, 0].set_title("Obstacle Map")
    axes[0, 0].axis("off")

    # 2. current agents
    axes[0, 1].imshow(obstacle_map, cmap="gray_r", alpha=0.35)
    axes[0, 1].imshow(agent_pos, cmap="Reds", alpha=0.8)
    axes[0, 1].set_title("Current Agents")
    axes[0, 1].axis("off")

    # 3. goals
    axes[0, 2].imshow(obstacle_map, cmap="gray_r", alpha=0.35)
    axes[0, 2].imshow(goal_pos, cmap="Greens", alpha=0.8)
    axes[0, 2].set_title("Goal Locations")
    axes[0, 2].axis("off")

    # 4. GT heatmap
    im_gt = axes[0, 3].imshow(gt_heatmap, cmap="hot", vmin=0, vmax=1)
    axes[0, 3].set_title("GT Rolling-window Heatmap")
    axes[0, 3].axis("off")
    plt.colorbar(im_gt, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # 5. predicted heatmap raw
    im_pred = axes[1, 0].imshow(pred_heatmap, cmap="hot", vmin=0, vmax=1)
    axes[1, 0].set_title("Predicted Heatmap")
    axes[1, 0].axis("off")
    plt.colorbar(im_pred, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 6. predicted heatmap cleaned
    im_clean = axes[1, 1].imshow(pred_heatmap_clean, cmap="hot", vmin=0, vmax=1)
    axes[1, 1].set_title("Predicted Heatmap > 0.15")
    axes[1, 1].axis("off")
    plt.colorbar(im_clean, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 7. error map
    im_err = axes[1, 2].imshow(error_heatmap, cmap="magma", vmin=0, vmax=1)
    axes[1, 2].set_title("Absolute Heatmap Error")
    axes[1, 2].axis("off")
    plt.colorbar(im_err, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # 8. action correctness
    correctness = torch.zeros_like(label, dtype=torch.float32)
    correctness[valid_mask] = (gt_action_show[valid_mask] == pred_action_show[valid_mask]).float()

    axes[1, 3].imshow(obstacle_map, cmap="gray_r", alpha=0.25)
    axes[1, 3].imshow(correctness, cmap="viridis", vmin=0, vmax=1)
    axes[1, 3].set_title("Action Correctness at Agent Cells")
    axes[1, 3].axis("off")

    # =========================
    # 指标
    # =========================
    if valid_mask.sum() > 0:
        action_acc = (
            gt_action_show[valid_mask] == pred_action_show[valid_mask]
        ).float().mean().item()
    else:
        action_acc = 0.0

    heatmap_mae = error_heatmap.mean().item()
    gt_nonzero = int((gt_heatmap > 0).sum().item())
    pred_nonzero = int((pred_heatmap_clean > 0).sum().item())

    fig.suptitle(
        f"Sample {sample_id} | "
        f"Action Acc={action_acc:.3f} | "
        f"Heatmap MAE={heatmap_mae:.4f} | "
        f"GT nonzero={gt_nonzero} | Pred>0.15 nonzero={pred_nonzero}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()

    print(f"Saved: {save_path}")
    print(
        f"Sample {sample_id}: "
        f"Action Acc={action_acc:.3f}, "
        f"Heatmap MAE={heatmap_mae:.4f}, "
        f"GT nonzero={gt_nonzero}, "
        f"Pred>0.15 nonzero={pred_nonzero}"
    )


# =========================
# 4. 主函数
# =========================
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Save dir: {SAVE_DIR}")

    dataset = MAPFDataset(
        data_dir=DATA_DIR,
        return_raw=False,
        strict=True,
        label_mode="all",
        clamp_heatmap=True,
    )

    print(f"Loaded {len(dataset)} validation samples.")

    model = load_model()

    indices = random.sample(range(len(dataset)), min(NUM_SAMPLES, len(dataset)))

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        save_path = os.path.join(SAVE_DIR, f"rhcr_heatmap_vis_{i:03d}_idx{idx}.png")
        visualize_one_sample(sample, model, save_path, sample_id=idx)

    print("\nDone.")
    print(f"All visualizations saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()