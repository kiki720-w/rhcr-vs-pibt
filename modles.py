import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. CBAM 注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 8):
        super().__init__()
        hidden = max(1, in_planes // ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size 建议用 3 或 7"
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAMBlock(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 8, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# 2. 基础残差块
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_cbam: bool = False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAMBlock(out_channels)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_cbam:
            out = self.cbam(out)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


# 3. 改良版 Three Encoders + ResUNet
class MAPF_ResUNet(nn.Module):
    def __init__(
        self,
        num_actions: int = 5,
        use_aux_head: bool = False,
        dropout_p: float = 0.10,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.use_aux_head = use_aux_head

        # ----------------------------------
        # [A] 三编码器
        # ----------------------------------
        self.map_enc = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ResBlock(16, 16, use_cbam=False),
        )

        self.agent_enc = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32, 32, use_cbam=False),
        )

        self.res_enc = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ResBlock(16, 16, use_cbam=False),
        )

        # ----------------------------------
        # [B] 融合层：更厚一点
        # ----------------------------------
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64, 64, use_cbam=False),
        )

        self.dropout = nn.Dropout2d(p=dropout_p)

        # ----------------------------------
        # [C] U-Net 主干
        # 32x32 -> 16x16 -> 8x8 -> 16x16 -> 32x32
        # ----------------------------------
        self.down1 = ResBlock(64, 128, use_cbam=False)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = ResBlock(128, 256, use_cbam=True)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ResBlock(256, 256, use_cbam=True)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv1 = ResBlock(384, 128, use_cbam=False)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv2 = ResBlock(192, 64, use_cbam=False)

        # ----------------------------------
        # [D] 输出头
        # ----------------------------------
        self.action_head = nn.Conv2d(64, num_actions, kernel_size=1)

        # 可选辅助头：预测拥堵/flow
        if self.use_aux_head:
            self.heatmap_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(
        self,
        map_feat: torch.Tensor,
        agent_feat: torch.Tensor,
        res_feat: torch.Tensor,
        return_aux: bool = False,
    ):
        # 1) 三编码器
        e_map = self.map_enc(map_feat)        # [B, 16, 32, 32]
        e_agent = self.agent_enc(agent_feat)  # [B, 32, 32, 32]
        e_res = self.res_enc(res_feat)        # [B, 16, 32, 32]

        # 2) 融合
        x = torch.cat([e_map, e_agent, e_res], dim=1)  # [B, 64, 32, 32]
        x = self.fusion(x)
        x = self.dropout(x)

        # 3) 下采样
        d1 = self.down1(x)     # [B, 128, 32, 32]
        p1 = self.pool1(d1)    # [B, 128, 16, 16]

        d2 = self.down2(p1)    # [B, 256, 16, 16]
        p2 = self.pool2(d2)    # [B, 256, 8, 8]

        # 4) bottleneck
        b = self.bottleneck(p2)  # [B, 256, 8, 8]

        # 5) 上采样 + skip
        u1 = self.up1(b)                     # [B, 128, 16, 16]
        u1_cat = torch.cat([u1, d2], dim=1) # [B, 256, 16, 16]
        u1_out = self.up_conv1(u1_cat)      # [B, 128, 16, 16]

        u2 = self.up2(u1_out)               # [B, 64, 32, 32]
        u2_cat = torch.cat([u2, d1], dim=1) # [B, 128, 32, 32]
        u2_out = self.up_conv2(u2_cat)      # [B, 64, 32, 32]

        # 6) 主输出
        action_logits = self.action_head(u2_out)  # [B, 5, 32, 32]

        # 默认只返回主任务，先把训练跑稳
        if not self.use_aux_head or not return_aux:
            return action_logits

        # 可选辅助输出
        heatmap_logits = self.heatmap_head(u2_out)
        return action_logits, heatmap_logits


# 4. 验机
if __name__ == "__main__":
    print("正在实例化改良版 MAPF_ResUNet...")

    # 推荐先单任务
    model = MAPF_ResUNet(num_actions=5, use_aux_head=False, dropout_p=0.10)

    dummy_map = torch.randn(4, 2, 32, 32)
    dummy_agent = torch.randn(4, 5, 32, 32)
    dummy_res = torch.randn(4, 2, 32, 32)

    print("喂入特征数据进行前向传播...")
    out_action = model(dummy_map, dummy_agent, dummy_res)

    print("\单任务前向传播测试通过！")
    print(f" Action 输出维度: {out_action.shape}")   # [4, 5, 32, 32]

    # 如果你想测试辅助头：
    model_aux = MAPF_ResUNet(num_actions=5, use_aux_head=True, dropout_p=0.10)
    out_action2, out_heatmap = model_aux(dummy_map, dummy_agent, dummy_res, return_aux=True)

    print("多任务前向传播测试通过！")
    print(f"Action 输出维度: {out_action2.shape}")
    print(f"Heatmap 输出维度: {out_heatmap.shape}")