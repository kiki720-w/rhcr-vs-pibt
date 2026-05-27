import os
import glob
from typing import Dict, Any

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class MAPFDataset(Dataset):
    """
    MAPF 多任务数据集加载器

    每个 .pt 文件应包含：
        {
            "map_x": Tensor[9, H, W],
            "label": Tensor[H, W],
            "heatmap_target": Tensor[H, W]
        }

    返回：
        {
            "map_feat": Tensor[2, H, W],
            "agent_feat": Tensor[5, H, W],
            "res_feat": Tensor[2, H, W],
            "label": Tensor[H, W],
            "heatmap_target": Tensor[1, H, W]
        }
    """

    F_MAP = 0
    F_CUR = 1
    F_GOAL = 2
    F_CG = 3
    F_GRAD_X = 4
    F_GRAD_Y = 5
    F_CAPACITY = 6
    F_TIME = 7
    F_FLOW = 8

    MAP_CHANNELS = [F_MAP, F_CAPACITY]
    AGENT_CHANNELS = [F_CUR, F_GOAL, F_CG, F_GRAD_X, F_GRAD_Y]
    RES_CHANNELS = [F_TIME, F_FLOW]

    def __init__(
        self,
        data_dir: str,
        return_raw: bool = False,
        strict: bool = True,
        label_mode: str = "downsample_stay",
        stay_keep_prob: float = 0.2,
        clamp_heatmap: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.return_raw = return_raw
        self.strict = strict
        self.label_mode = label_mode
        self.stay_keep_prob = stay_keep_prob
        self.clamp_heatmap = clamp_heatmap

        if label_mode not in {"all", "ignore_stay", "downsample_stay"}:
            raise ValueError(f"不支持的 label_mode: {label_mode}")

        if not (0.0 <= stay_keep_prob <= 1.0):
            raise ValueError("stay_keep_prob 必须在 [0, 1] 之间")

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.pt")))

        if len(self.file_list) == 0:
            raise FileNotFoundError(
                f"在目录 {data_dir} 中没有找到任何 .pt 文件，请检查路径。"
            )

    def __len__(self) -> int:
        return len(self.file_list)

    def _validate_sample(self, data: Dict[str, Any], file_path: str) -> None:
        required_keys = ["map_x", "label", "heatmap_target"]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"{file_path} 缺少 '{key}' 键。")

        map_x = data["map_x"]
        label = data["label"]
        heatmap_target = data["heatmap_target"]

        if not isinstance(map_x, torch.Tensor):
            raise TypeError(f"{file_path} 中 map_x 不是 torch.Tensor。")
        if not isinstance(label, torch.Tensor):
            raise TypeError(f"{file_path} 中 label 不是 torch.Tensor。")
        if not isinstance(heatmap_target, torch.Tensor):
            raise TypeError(f"{file_path} 中 heatmap_target 不是 torch.Tensor。")

        if map_x.ndim != 3:
            raise ValueError(f"{file_path} 中 map_x 应为 [9,H,W]，实际 {tuple(map_x.shape)}")
        if label.ndim != 2:
            raise ValueError(f"{file_path} 中 label 应为 [H,W]，实际 {tuple(label.shape)}")
        if heatmap_target.ndim != 2:
            raise ValueError(
                f"{file_path} 中 heatmap_target 应为 [H,W]，实际 {tuple(heatmap_target.shape)}"
            )

        if map_x.shape[0] != 9:
            raise ValueError(f"{file_path} 中 map_x 通道数应为 9，实际 {map_x.shape[0]}")

        if map_x.shape[1:] != label.shape:
            raise ValueError(
                f"{file_path} 中 map_x 空间尺寸 {tuple(map_x.shape[1:])} "
                f"与 label 尺寸 {tuple(label.shape)} 不一致。"
            )

        if map_x.shape[1:] != heatmap_target.shape:
            raise ValueError(
                f"{file_path} 中 map_x 空间尺寸 {tuple(map_x.shape[1:])} "
                f"与 heatmap_target 尺寸 {tuple(heatmap_target.shape)} 不一致。"
            )

    def _process_label(self, label: Tensor) -> Tensor:
        label = label.clone()

        if self.label_mode == "all":
            return label

        if self.label_mode == "ignore_stay":
            label[label == 0] = -1
            return label

        if self.label_mode == "downsample_stay":
            stay_mask = (label == 0)
            if stay_mask.any():
                rand_mask = torch.rand(label.shape)
                drop_mask = stay_mask & (rand_mask > self.stay_keep_prob)
                label[drop_mask] = -1
            return label

        return label

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        file_path = self.file_list[idx]

        data = torch.load(file_path, map_location="cpu", weights_only=False)

        if self.strict:
            self._validate_sample(data, file_path)

        map_x: Tensor = data["map_x"].float()
        label: Tensor = data["label"].long()
        heatmap_target: Tensor = data["heatmap_target"].float()

        label = self._process_label(label)

        if self.clamp_heatmap:
            heatmap_target = torch.clamp(heatmap_target, 0.0, 1.0)

        # [H, W] -> [1, H, W]，方便和模型输出 [B,1,H,W] 对齐
        heatmap_target = heatmap_target.unsqueeze(0)

        map_feat = map_x[self.MAP_CHANNELS, :, :]
        agent_feat = map_x[self.AGENT_CHANNELS, :, :]
        res_feat = map_x[self.RES_CHANNELS, :, :]

        sample = {
            "map_feat": map_feat,
            "agent_feat": agent_feat,
            "res_feat": res_feat,
            "label": label,
            "heatmap_target": heatmap_target,
        }

        if self.return_raw:
            sample["map_x"] = map_x

        return sample


def build_dataloader(
    data_dir: str,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    return_raw: bool = False,
    strict: bool = True,
    label_mode: str = "downsample_stay",
    stay_keep_prob: float = 0.2,
    clamp_heatmap: bool = True,
) -> DataLoader:
    dataset = MAPFDataset(
        data_dir=data_dir,
        return_raw=return_raw,
        strict=strict,
        label_mode=label_mode,
        stay_keep_prob=stay_keep_prob,
        clamp_heatmap=clamp_heatmap,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


if __name__ == "__main__":
    test_dir = "./dataset_v2_random/val"

    print(f"正在加载数据集: {test_dir}")

    dataset = MAPFDataset(
        data_dir=test_dir,
        return_raw=False,
        strict=True,
        label_mode="downsample_stay",
        stay_keep_prob=0.2,
        clamp_heatmap=True,
    )

    print(f"成功找到 {len(dataset)} 个样本。")

    dataloader = build_dataloader(
        data_dir=test_dir,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        return_raw=False,
        strict=True,
        label_mode="downsample_stay",
        stay_keep_prob=0.2,
        clamp_heatmap=True,
    )

    for batch in dataloader:
        print("\n📦 抽取到一个 batch：")
        print(f"map_feat       : {batch['map_feat'].shape}")
        print(f"agent_feat     : {batch['agent_feat'].shape}")
        print(f"res_feat       : {batch['res_feat'].shape}")
        print(f"label          : {batch['label'].shape}")
        print(f"heatmap_target : {batch['heatmap_target'].shape}")

        valid_labels = batch["label"][batch["label"] != -1]
        print(f"\n有效动作标签数量: {valid_labels.numel()}")

        if valid_labels.numel() > 0:
            print(f"前 15 个有效动作标签: {valid_labels[:15].tolist()}")

            unique_vals, counts = torch.unique(valid_labels, return_counts=True)
            print("当前 batch 动作标签分布:")
            for v, c in zip(unique_vals.tolist(), counts.tolist()):
                print(f"  label={v}: {c}")

        heatmap = batch["heatmap_target"]
        print("\nheatmap_target 检查:")
        print(f"  min: {heatmap.min().item():.4f}")
        print(f"  max: {heatmap.max().item():.4f}")
        print(f"  mean: {heatmap.mean().item():.4f}")
        print(f"  nonzero: {(heatmap > 0).sum().item()}")

        break