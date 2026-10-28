"""OOD 背景源 —— 对应参考实现的 `robosaga/background_randomizer.py`。

参考实现从 `backgrounds/` 随机取图，做「随机旋转 0–360° + 亮度 ±0.1」，
再用**与观测同一个 normalizer** 归一化 —— 因为混合是在归一化空间里做的。
这里照做，但有两处必须说明的差别。

## 差别 1：旋转不实现（因为它是恒等变换）

用户 2026-09-22 定的是**两臂统一用 `backgrounds/solid8` 纯色**，不用仓库那 5908 张。
那 8 张每张**只有一个唯一颜色**（64×64，实测唯一值数 = 1）。
对常量图像做任何角度的旋转都是恒等变换，写这段代码是纯粹的仪式。
（若将来换成真实照片池，这一行必须补上。）

## 差别 2：亮度抖动的加/乘语义没核对上

参考实现用 kornia 的 `K.enhance.adjust_brightness`，本机没装 kornia，
它不同版本里是**加性**（`img + f`）还是**乘性**（`img * f`）各有其例。这里取**加性**。
±0.1 的幅度在两种语义下都只是很小的一点颜色偏移，对「纯色 vs 纯色」的对照不构成
方向性影响；但要完全对齐论文请先核对 kornia 版本。
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

# LeRobot 在 `datasets/factory.py` 里把相机统计量覆写成 ImageNet 的，
# 所以 batch 里的图就是这个空间。背景必须落在同一空间才能混。
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """与 LeRobot `NormalizerProcessorStep` 对图像做的变换一致。"""
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


class BackgroundRandomizer:
    """每批随机抽 N 张 OOD 背景，返回**归一化后**的 (N,3,H,W)。"""

    def __init__(self, background_path: str, output_shape: tuple[int, int],
                 brightness_jitter: float = 0.1, device="cuda", verbose=True):
        self.im_size = (int(output_shape[0]), int(output_shape[1]))
        self.brightness_jitter = float(brightness_jitter)
        self.device = device
        self.paths = sorted(
            p for p in glob.glob(os.path.join(background_path, "*"))
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        )
        if not self.paths:
            raise SystemExit(
                f"{background_path} 里没有图片。SaGA 需要一个 OOD 池。\n"
                f"先跑：python workplace/background_generalization/scripts/make_backgrounds.py "
                f"--mode solid --n 8 --out workplace/background_generalization/backgrounds/solid8"
            )
        self.backgrounds = self._preload()
        if verbose:
            print(f"[bg] {len(self.paths)} 张 @ {background_path} "
                  f"→ {self.im_size}  亮度抖动 ±{self.brightness_jitter}（无旋转：纯色下是恒等）")

    def _preload(self) -> torch.Tensor:
        from PIL import Image
        arrs = []
        for p in self.paths:
            im = Image.open(p).convert("RGB")
            a = np.asarray(im).astype(np.float32) / 255.0     # (h,w,3)
            arrs.append(torch.from_numpy(a).permute(2, 0, 1))  # (3,h,w)
        bg = torch.stack(arrs)
        if tuple(bg.shape[-2:]) != self.im_size:
            bg = F.interpolate(bg, size=self.im_size, mode="bilinear", align_corners=False)
        return bg.to(self.device)

    def __call__(self, num_samples: int) -> torch.Tensor:
        idx = torch.randint(0, self.backgrounds.shape[0], (num_samples,),
                            device=self.device)
        bg = self.backgrounds[idx].clone()

        if self.brightness_jitter > 0:
            f = (torch.rand(num_samples, 1, 1, 1, device=self.device) * 2 - 1) \
                * self.brightness_jitter
            bg = (bg + f).clamp(0, 1)

        return normalize_imagenet(bg)

    def __str__(self):
        return (f"BackgroundRandomizer({len(self.paths)} backgrounds, "
                f"size {self.im_size})")
