"""全局显著度缓冲区 —— 移植自 RoboSaGA 的 `robosaga/buffer_manager.py`。

论文（Algorithm 1 第 4-6 行）要求它按**全局 id** 存取显著图，并按「更新次数最少」
挑选每批更新的那 β 比例。原实现是 uint8 单通道、初值 255（即 1.0）。

## 与参考实现的一处简化

原实现存的是 `crop_idx` 画的画布 —— 因为它的图像管线里有 Random Crop，需要把
裁剪后的显著图放回原图坐标。我们**不加 Random Crop**（用户 2026-09-22 决定），
所以不需要 `crop_idx`，直接按尺寸缩放即可。

## 分辨率：这是唯一要权衡的地方

参考实现是 84×84 的图存 84×84 的缓冲。我们是 480×640：

    21998 帧 × 480 × 640 × 1 B ≈ 6.6 GB / 进程

两张 2080Ti 各 11 GB，训练时还要放模型和激活值 —— 会 OOM。
论文自己也写了缓冲区是 "at a reduced resolution"，所以降采样是**符合原意**的做法。

默认 120×160（1/4 边长）→ 约 422 MB。`buffer_shape` 可配。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class BufferManager:
    def __init__(self, buffer_depth: int, buffer_shape: tuple[int, int],
                 output_shape: tuple[int, int], device="cuda", verbose=True):
        """
        Args:
            buffer_depth: 缓冲区容量，= 数据集总帧数 + 1（全局 id 空间）。
            buffer_shape: 缓冲里存的显著图尺寸 (H, W)。
            output_shape: 取回时要缩放到 (H, W)，即真实图像尺寸。
        """
        self.buffer_depth = int(buffer_depth)
        self.buffer_shape = tuple(int(v) for v in buffer_shape)
        self.output_shape = tuple(int(v) for v in output_shape)
        self.device = device

        self.update_counter = torch.zeros(self.buffer_depth, device=device)
        H, W = self.buffer_shape
        # 初值 255 —— 即 s=1，等价于「还没算过显著度的地方保留原图」
        self.buffer = torch.ones(
            self.buffer_depth, 1, H, W, device=device
        ).mul_(255).to(torch.uint8)

        if verbose:
            mb = self.buffer.numel() / 1024**2
            print(f"[buffer] {self.buffer_depth} 帧 × {H}×{W} uint8 = {mb:.0f} MB "
                  f"(device={device})，取回时缩放到 {self.output_shape}")

    def set(self, smaps: torch.Tensor, buffer_ids: torch.Tensor):
        """存入一批显著图。

        Args:
            smaps: (B, 1, h, w) 或 (B, h, w)，取值范围必须已在 [0, 1]。
            buffer_ids: (B,) 全局 id。
        """
        if smaps.dim() == 3:
            smaps = smaps.unsqueeze(1)
        assert smaps.shape[0] == buffer_ids.shape[0], "显著图与 id 数量不一致"
        assert smaps.min() >= 0 and smaps.max() <= 1, (
            f"显著图不在 [0, 1] 内：min={smaps.min():.4f} max={smaps.max():.4f}"
        )

        # 同一批里同一个 id 出现多次时只更新第一次，避免重复计数
        uniq = self._first_occurrence_indices(buffer_ids)
        buffer_ids = buffer_ids[uniq]
        smaps = smaps[uniq]

        if tuple(smaps.shape[-2:]) != self.buffer_shape:
            smaps = F.interpolate(
                smaps.float(), size=self.buffer_shape, mode="bilinear", align_corners=False
            )
        smaps = (smaps * 255).to(torch.uint8)

        self.buffer[buffer_ids] = smaps.to(torch.uint8)
        self.update_counter[buffer_ids] += 1

    def get(self, buffer_ids: torch.Tensor) -> torch.Tensor:
        """取回一批显著图，返回 (B, 1, H_out, W_out)，值域 [0, 1]。"""
        smaps = self.buffer[buffer_ids] / 255.0
        if tuple(smaps.shape[-2:]) != self.output_shape:
            smaps = F.interpolate(
                smaps.float(), size=self.output_shape, mode="bilinear", align_corners=False
            )
        return smaps

    @staticmethod
    def _first_occurrence_indices(buffer_ids: torch.Tensor) -> torch.Tensor:
        id_dict: dict[int, int] = {}
        for i, id_ in enumerate(buffer_ids.tolist()):
            if id_ not in id_dict:
                id_dict[id_] = i
        return torch.tensor([id_dict[id_] for id_ in buffer_ids.tolist()])
