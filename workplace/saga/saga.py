"""RoboSaGA 驱动 —— 照论文 Algorithm 1 与参考实现 `robosaga/saliency_guided_augmentation.py`。

流程（论文 Algorithm 1，第 13-19 行）：

    # 更新步
    按缓冲区更新次数排序全局 id，取最久未更新的 β 比例
    new_maps = clip(normalize(FullGrad(encoder(imgs[Iβ]))), 0, λ)
    buffer.set(new_maps, Iβ)

    # 增强步
    Iα = 前 α 比例
    Sα = buffer.get(Iα)
    imgs[Iα] = Sα ⊙ imgs[Iα] + (1 − Sα) ⊙ ood_images()

## 三处必须记牢的顺序细节（都读自源码，论文正文没全写）

1. **缓冲区存的是归一化后、未裁剪的图**；`clip(·, λ)` 在**取回时**做
   （`extract_and_update` vs `retrieve_smaps_from_buffer`）。
2. 取回并裁剪之后，还有一步 **`s = s²`**（`saliency_guided_augmentation.py:137`）——
   论文正文里没有这一步，但代码里有。平方让 s 更小 → 更多 OOD 覆盖。
3. **混合在归一化空间做**：`bg` 用同一套 ImageNet 统计量归一化后才参与混合。

## 两处刻意偏离参考实现（不改变数学，理由写在各处）

- 钩子只在算显著度的那一次前向里注册，算完立刻摘掉。
- 不原地改传入的 batch 里的非图像键。
"""

from __future__ import annotations

import torch

from .backgrounds import BackgroundRandomizer
from .buffer import BufferManager
from .fullgrad import FullGrad, EncoderOnly


def normalize_smaps(smaps: torch.Tensor) -> torch.Tensor:
    """逐样本 min-max 归一化到 [0,1]。对应 `vis_utils.normalize_smaps`。"""
    if smaps.dim() == 4:
        smaps = smaps.squeeze(1)
    s_min = torch.amin(smaps, dim=(1, 2), keepdim=True)
    s_max = torch.amax(smaps, dim=(1, 2), keepdim=True)
    return torch.clamp((smaps - s_min) / (s_max - s_min + 1e-6), 0, 1)


class RoboSaGA:
    def __init__(
        self,
        backbone: torch.nn.Module,
        image_keys: list[str],
        image_shape: tuple[int, int],
        background_path: str,
        buffer_depth: int,
        aug_strategy: str = "robosaga",
        aug_ratio: float = 0.5,
        update_ratio: float = 0.1,
        saliency_saga_cap: float = 0.8,
        blend_alpha: float = 0.5,
        warmup_steps: int = 0,
        buffer_shape: tuple[int, int] = (120, 160),
        disable_buffer: bool = False,
        device="cuda",
        debug_dir: str | None = None,
        debug_every: int = 50,
    ):
        assert aug_strategy in ("robosaga", "random_overlay"), aug_strategy
        assert 0.0 < saliency_saga_cap < 1.0, "λ 必须在 (0,1)"

        self.backbone = backbone
        self.image_keys = list(image_keys)
        self.image_shape = tuple(image_shape)
        self.aug_strategy = aug_strategy
        self.aug_ratio = float(aug_ratio)
        self.update_ratio = float(update_ratio)
        self.saliency_saga_cap = float(saliency_saga_cap)
        self.blend_alpha = float(blend_alpha)
        self.warmup_steps = int(warmup_steps)
        self.disable_buffer = bool(disable_buffer)
        self.device = device
        self.debug_dir = debug_dir
        self.debug_every = int(debug_every)
        self._warned_zero_update = False

        self.backgrounds = BackgroundRandomizer(
            background_path, self.image_shape, device=device
        )
        self.fullgrad = FullGrad(backbone, obs_key=None, Extractor=EncoderOnly)
        # FullGrad 在构造时就把 forward hook 挂上了。这里立刻摘掉 —— 训练主前向
        # （policy.forward 那一遍）绝不该为显著度付任何开销。真正需要时由
        # `_fullgrad` 现场注册、用完即摘。
        self.fullgrad.unregister_hooks()
        self.buffers: dict[str, BufferManager] = {}
        if not self.disable_buffer:
            for k in self.image_keys:
                self.buffers[k] = BufferManager(
                    buffer_depth=buffer_depth,
                    buffer_shape=buffer_shape,
                    output_shape=self.image_shape,
                    device=device,
                )
        print(f"[saga] strategy={aug_strategy} α={aug_ratio} β={update_ratio} "
              f"λ={saliency_saga_cap} warmup={warmup_steps} 步  "
              f"相机={[k.split('.')[-1] for k in self.image_keys]}  "
              f"buffer={'off' if disable_buffer else 'on'}")
        if aug_strategy == "robosaga" and disable_buffer:
            print("[saga][!!] robosaga 但缓冲区关掉了 —— 这条路径参考实现里**不做 λ 裁剪**，"
                  "且只更新不检索。这是消融臂，不是主臂。")

    # ------------------------------------------------------------------ #
    #                              主入口                                 #
    # ------------------------------------------------------------------ #

    def __call__(self, batch: dict, step: int) -> dict:
        """在 `preprocessor` **之后**、`policy.forward` **之前**调用。

        返回新的 batch（只替换被增强的相机图像键，其余键原样引用）。
        """
        # 红线第三条「宁可漏增强，不要错增强」的守门：这些情况一律原样返回
        if step < self.warmup_steps or self.aug_ratio <= 0:
            return batch
        if "index" not in batch:
            raise RuntimeError(
                "batch 里没有 'index'，拿不到全局 id → 无法索引显著度缓冲区。"
                "它本该由 convertors._extract_complementary_data 保留下来。"
            )

        gid = batch["index"]
        out = dict(batch)
        for key in self.image_keys:
            imgs = batch.get(key)
            if imgs is None:
                continue
            if self.aug_strategy == "random_overlay":
                out[key] = self._random_overlay(imgs)
            else:
                out[key] = self._saga(imgs, gid, self.buffers[key])
        return out

    # ------------------------------------------------------------------ #
    #                            两种增强策略                             #
    # ------------------------------------------------------------------ #

    def _random_overlay(self, imgs: torch.Tensor) -> torch.Tensor:
        """论文的头号对照：全局常数混合系数。参考实现 `random_overlay()`。"""
        n = imgs.shape[0]
        k = int(n * self.aug_ratio)
        if k == 0:
            return imgs
        bg = self.backgrounds(k)
        out = imgs.clone()
        a = self.blend_alpha
        out[:k] = imgs[:k] * a + bg * (1 - a)
        return out

    def _update_batch_idx(self, gid: torch.Tensor, buffer: BufferManager,
                          imgs: torch.Tensor) -> torch.Tensor:
        """挑出本批要更新显著度的 β 比例。照 `get_update_batch_idx` 逐行。

        注意参考实现里那个 **关掉缓冲区就改随机** 的分支（`mode = "random" if
        self.disable_buffer else mode`）—— 不照抄的话消融臂量的会是另一个东西。
        """
        n = len(gid)
        n_augs = int(n * max(self.aug_ratio, self.update_ratio))
        n_updates = min(n_augs, int(n * self.update_ratio))
        if n_updates == 0:
            return gid[:0]
        if self.disable_buffer:
            return torch.randperm(n, device=imgs.device)[:n_updates]
        freq = buffer.update_counter[gid]
        _, order = torch.sort(freq)
        return order[:n_updates]

    def _saga(self, imgs: torch.Tensor, gid: torch.Tensor,
              buffer: BufferManager) -> torch.Tensor:
        n = imgs.shape[0]

        # ---- 更新步 ----
        # `imgs[upd_idx]` 高级索引本就是拷贝；再 clone 一次是防 FullGrad 内部那句
        # 就地的 `requires_grad_()` 把 batch 里的张量标脏。
        upd_idx = self._update_batch_idx(gid, buffer, imgs)
        if len(upd_idx) == 0 and not self._warned_zero_update:
            self._warned_zero_update = True
            print(
                f"[saga][!!] int(batch_size × β) = int({n} × {self.update_ratio}) = 0 —— "
                f"本批不更新任何显著图。缓冲区会**一直停在初值 1.0**，"
                f"于是 s ≡ clip(1,0,λ)² = {self.saliency_saga_cap ** 2:.3f} 是个常数，\n"
                f"           SaGA 实际退化成「random_overlay + 固定权重」，显著度引导完全没生效。\n"
                f"           要真正跑 SaGA，batch_size 必须 ≥ 1/β = {1 / self.update_ratio:.0f}。"
            )
        # 参考实现无条件调 extract_and_update；这里加 len 判断只是防 0 元素前向崩掉。
        smaps_new = (normalize_smaps(self._fullgrad(imgs[upd_idx].clone())).detach()
                     if len(upd_idx) else None)
        if not self.disable_buffer and smaps_new is not None:
            buffer.set(smaps_new, gid[upd_idx])

        # ---- 增强步 ----
        if self.disable_buffer:
            # 关缓冲区时：增强集 == 更新集，且用刚算出的图、**不做 λ 裁剪**。
            # 这是参考实现的怪癖（它那条路径不经过 retrieve_smaps_from_buffer），
            # 照抄 —— 否则消融臂量的不是同一个东西。
            aug_idx, smaps = upd_idx, smaps_new
        else:
            k = int(n * self.aug_ratio)
            if k == 0:
                return imgs
            aug_idx = torch.arange(k, device=imgs.device)
            smaps = buffer.get(gid[aug_idx])
            if self.aug_strategy == "robosaga":
                smaps = torch.clip(smaps, 0, self.saliency_saga_cap)

        if smaps is None or len(aug_idx) == 0:
            return imgs

        # 源码里论文正文没写的一步
        smaps = smaps ** 2

        bg = self.backgrounds(len(aug_idx))
        out = imgs.clone()
        out[aug_idx] = imgs[aug_idx] * smaps + bg * (1 - smaps)
        return out

    def _fullgrad(self, x: torch.Tensor) -> torch.Tensor:
        """算显著度。钩子只在这一次前向里注册，算完立刻摘掉。

        偏离参考实现之处：它把钩子一直挂着（整轮训练都注册着）。那样训练的反传也会
        触发钩子、把训练图的张量塞进 `feature_grads` 并活到下次调用 —— 白占一份显存。
        此处改为用完即摘，数学完全不变。
        """
        was_training = self.backbone.training
        # 参考实现强制 eval。对 ACT 的 resnet18 其实是个 no-op
        # （FrozenBatchNorm2d 的 train/eval 前向逐位相同），但保留以对齐。
        self.backbone.eval()
        self.fullgrad.register_hooks()
        try:
            return self.fullgrad(x)
        finally:
            self.fullgrad.unregister_hooks()
            self.backbone.train(was_training)
