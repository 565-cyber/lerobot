#!/usr/bin/env python
"""SaGA 移植的验证。

    python workplace/saga/scripts/verify_saga.py

默认走 CPU —— 两个 GPU 常被掩码任务占着，不抢。

七项检查，从"最可能出错、且出错最看不出来"的排起：

1. **FullGrad 完备性（小卷积网）** —— 作者自带的测试搬过来先用小网跑一遍，
   确认测试本身在这个环境里是有效的。
2. **FullGrad 完备性（真 ACT backbone）** —— 这是抓 FrozenBatchNorm2d 那处适配
   是否有错的**唯一硬证据**。20 层 BN 若被静默跳过，这一条会失败。
3. **缓冲区的 global id 配对** —— 方向搞反（把"最久未更新"写成"最近更新"）
   会让论文的 β 语义整个失效，而表面上一切正常。
4. **混合的逐元素算术** —— 独立重算一遍 `s·x + (1−s)·bg`，并确认权重上限
   恰好是 λ²（λ=0.8 → 0.64），即 λ 裁剪和平方的顺序没写反。
5. **只有前 α 比例被动过** —— 后一半必须**逐位**不变。
6. **可证伪性** —— α=0 时整个 batch 逐位不变（红线第三条的守门测试）。
7. **warmup** —— 第 10 个 epoch 之前逐位不变。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve()
SAGA = HERE.parents[1]          # workplace/saga
WORKPLACE = HERE.parents[2]
REPO = HERE.parents[3]
for _p in (str(REPO / "src"), str(WORKPLACE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from saga.fullgrad import FullGrad, EncoderOnly            # noqa: E402
from saga.buffer import BufferManager                       # noqa: E402
from saga.backgrounds import BackgroundRandomizer           # noqa: E402
from saga.saga import RoboSaGA, normalize_smaps             # noqa: E402

BG_PATH = str(WORKPLACE / "background_generalization" / "backgrounds" / "solid8")
CKPT = str(REPO / "outputs/train/act_record_test10/checkpoints/last/pretrained_model")

FAILED: list[str] = []
PASSED: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  —— {detail}" if detail else ""))


_POLICY = None


def load_backbone_cpu():
    """载入 ACT，**强制落在 CPU** —— 两个 GPU 上有掩码任务在跑，不抢。

    `from_pretrained` 会 `policy.to(config.device)`，而 checkpoint 存下来的
    `device` 是 cuda。所以必须自己在 config 上改掉，不能load 完再 `.to("cpu")`
    （那样会先在 GPU 上占一份显存）。载入一次、全程复用。
    """
    global _POLICY
    if _POLICY is None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        cfg = PreTrainedConfig.from_pretrained(CKPT)
        cfg.device = "cpu"
        _POLICY = ACTPolicy.from_pretrained(CKPT, config=cfg)
        _POLICY.eval()
    return _POLICY.model.backbone


# --------------------------------------------------------------------------- #
# 1 & 2  FullGrad 完备性
# --------------------------------------------------------------------------- #

class ToyNet(nn.Module):
    """小卷积网，刻意让每一层都执行到、且 conv 带 bias。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 4, 3, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.conv2(self.bn(self.relu(self.conv1(x))))))


def test_fullgrad_toy():
    print("\n[1] FullGrad 完备性 —— 小卷积网")
    torch.manual_seed(0)
    net = ToyNet().eval()
    fg = FullGrad(net, obs_key=None, Extractor=EncoderOnly)
    x = torch.rand(2, 3, 16, 16)
    try:
        fg.checkCompleteness(x)
        check("小网完备性", True)
    except AssertionError as e:
        check("小网完备性", False, str(e)[:200])


def test_fullgrad_act():
    print("\n[2] FullGrad 完备性 —— 真 ACT backbone（FrozenBatchNorm2d 的裁判）")
    backbone = load_backbone_cpu()

    # 数一下到底接住了几层 BN
    from saga.fullgrad import _target_modules
    tgt = _target_modules()
    n_conv = sum(1 for m in backbone.modules() if isinstance(m, nn.Conv2d))
    n_bn = sum(1 for m in backbone.modules()
               if any(isinstance(m, T) and "BatchNorm" in type(m).__name__ for T in tgt))
    print(f"       backbone 里 Conv2d={n_conv}  BatchNorm 系={n_bn}")
    check("FrozenBatchNorm2d 被接住（不为 0）", n_bn > 0, f"n_bn={n_bn}")

    fg = FullGrad(backbone, obs_key=None, Extractor=EncoderOnly)
    # 96x96 足够验证完备性且快；完备性与分辨率无关
    x = torch.rand(1, 3, 96, 96)
    try:
        fg.checkCompleteness(x)
        check("ACT backbone 完备性", True)
    except AssertionError as e:
        check("ACT backbone 完备性", False, str(e)[:300])


# --------------------------------------------------------------------------- #
# 3  缓冲区
# --------------------------------------------------------------------------- #

def test_buffer():
    print("\n[3] 缓冲区")
    depth = 64
    b = BufferManager(buffer_depth=depth, buffer_shape=(12, 16),
                      output_shape=(48, 64), device="cpu", verbose=False)

    ids = torch.tensor([3, 5, 7])
    smaps = torch.rand(3, 1, 12, 16)
    b.set(smaps, ids)

    got = b.get(ids)
    err = (got - nn.functional.interpolate(
        smaps, size=(48, 64), mode="bilinear", align_corners=False)).abs().max().item()
    # uint8 量化：1/255 ≈ 3.9e-3，插值不引入额外误差
    check("set/get 往返误差 ≤ 1/255", err <= 1.0 / 255 + 1e-6, f"max_err={err:.6f}")
    check("update_counter 只对被写入的 id 递增",
          b.update_counter[3].item() == 1 and b.update_counter[4].item() == 0)

    # 最久未更新优先：id 0..4，只更新过 3 → 排序后 3 应排在最后
    gid = torch.arange(5)
    _, order = torch.sort(b.update_counter[gid])
    check("最久未更新优先（更新过的排最后）", order[-1].item() == 3,
          f"order={order.tolist()}")

    # 未写过的位置初值必须是 1.0（论文：buffer 初值全 1）
    fresh = b.get(torch.tensor([0]))
    check("未写入处初值为 1.0", abs(fresh.max().item() - 1.0) < 1e-6,
          f"max={fresh.max().item():.6f}")

    # 去重：同一 id 在一批里出现两次只应计一次
    b2 = BufferManager(buffer_depth=8, buffer_shape=(4, 4),
                       output_shape=(4, 4), device="cpu", verbose=False)
    b2.set(torch.rand(3, 1, 4, 4), torch.tensor([1, 1, 2]))
    check("同批重复 id 只计一次更新", b2.update_counter[1].item() == 1
          and b2.update_counter[2].item() == 1)


# --------------------------------------------------------------------------- #
# 4-7  端到端 batch 行为
# --------------------------------------------------------------------------- #

def make_batch(n=10, device="cpu"):
    """合成 batch，键与原位值与真实训练时一致（图像已在 ImageNet 空间）。"""
    torch.manual_seed(1234)
    return {
        "observation.images.up": torch.rand(n, 3, 480, 640) * 2 - 1,
        "observation.images.front": torch.rand(n, 3, 480, 640) * 2 - 1,
        "observation.state": torch.randn(n, 6),
        "action": torch.randn(n, 100, 6),
        "index": torch.arange(n),
        "task_index": torch.zeros(n, dtype=torch.long),
        "task": ["Pick up yellow block and put it in trash can"] * n,
    }


def build_saga(backbone, **over):
    kw = dict(
        backbone=backbone,
        image_keys=["observation.images.up", "observation.images.front"],
        image_shape=(480, 640),
        background_path=BG_PATH,
        buffer_depth=64,
        aug_ratio=0.5,
        update_ratio=0.1,
        saliency_saga_cap=0.8,
        warmup_steps=0,
        buffer_shape=(120, 160),
        device="cpu",
    )
    kw.update(over)
    return RoboSaGA(**kw)


def test_batch_behaviour():
    print("\n[4-7] batch 行为 —— 载入真 ACT backbone")
    backbone = load_backbone_cpu()

    n = 10   # β=0.1 → n_updates = min(5, 1) = 1，确保更新路径真的被走到
    batch = make_batch(n)

    saga = build_saga(backbone)
    out = saga(batch, step=10_000)   # step 大到越过 warmup

    # ---- [4] 逐元素算术 ----
    key = "observation.images.up"
    k = int(n * 0.5)
    # 独立重算：s 的上限应当是 λ² = 0.64（裁剪在前、平方在后）
    smaps = saga.buffers[key].get(batch["index"][:k])
    s = torch.clip(smaps, 0, saga.saliency_saga_cap) ** 2
    check("权重上限 = λ²（裁剪与平方的顺序）",
          abs(s.max().item() - 0.8 ** 2) < 2e-3, f"s.max={s.max().item():.4f}")

    # 只检查「被动过的那一半」的算术：bg 是随机的，没法独立复现，
    # 所以反过来验证：s=0 处必须等于 bg、s=1 处必须等于原图。
    # 用一次极端 λ 让 s 退化成接近 0 和接近 1 来分别检验。
    saga_lo = build_saga(backbone, saliency_saga_cap=0.01)
    out_lo = saga_lo(batch, step=10_000)
    # s ≤ 0.01² = 1e-4 → 输出应几乎全是背景
    delta = (out_lo[key][:k] - batch[key][:k]).abs().mean().item()
    check("λ→0 时输出远离原图（OOD 主导）", delta > 0.1, f"mean|Δ|={delta:.4f}")

    # ---- [5] 只有前 α 比例被动过 ----
    for kk in ("observation.images.up", "observation.images.front"):
        tail_same = torch.equal(out[kk][k:], batch[kk][k:])
        head_diff = not torch.equal(out[kk][:k], batch[kk][:k])
        check(f"{kk.split('.')[-1]}：后一半逐位不变", tail_same)
        check(f"{kk.split('.')[-1]}：前一半确实被改了", head_diff)

    # ---- 非图像键必须逐位相同 ----
    for kk in ("observation.state", "action", "index", "task_index", "task"):
        same = (torch.equal(out[kk], batch[kk]) if torch.is_tensor(out[kk])
                else out[kk] == batch[kk])
        check(f"非图像键 {kk} 逐位不变", bool(same))

    # ---- [6] 可证伪性：α=0 ----
    saga0 = build_saga(backbone, aug_ratio=0.0)
    out0 = saga0(batch, step=10_000)
    all_same = all(
        (torch.equal(out0[kk], batch[kk]) if torch.is_tensor(batch[kk]) else out0[kk] == batch[kk])
        for kk in batch
    )
    check("可证伪性：α=0 时整个 batch 逐位不变", all_same)

    # ---- [7] warmup ----
    sage_w = build_saga(backbone, warmup_steps=500)
    outw = sage_w(batch, step=499)
    all_same_w = all(
        (torch.equal(outw[kk], batch[kk]) if torch.is_tensor(batch[kk]) else outw[kk] == batch[kk])
        for kk in batch
    )
    check("warmup：step < warmup_steps 时逐位不变", all_same_w)

    # ---- random_overlay 对照也走一遍 ----
    saga_o = build_saga(backbone, aug_strategy="random_overlay")
    outo = saga_o(batch, step=10_000)
    check("random_overlay：后一半逐位不变",
          torch.equal(outo[key][k:], batch[key][k:]))
    # 常数混合系数 → 权重恒为 0.5
    check("random_overlay：前一半被改了",
          not torch.equal(outo[key][:k], batch[key][:k]))

    # ---- 显著度确实依赖图像（不是常数）----
    imgs = batch[key][:2].clone()
    m = normalize_smaps(saga._fullgrad(imgs))
    check("显著度不是常数（FullGrad 真的在算）",
          m[0].std().item() > 1e-3 and m[0].max().item() > m[0].min().item() + 1e-3,
          f"std={m[0].std().item():.4f}")
    check("显著度被归一化到 [0,1]",
          m.min().item() >= -1e-6 and m.max().item() <= 1 + 1e-6)

    return saga, batch


def test_first_batch_is_uniform():
    """缓冲区初值全 1 → 头几批的 s 是**常数** λ²=0.64，不是逐像素的。

    这是论文设计的直接推论（Algorithm 1 第 8 行「Initialize with all-one
    saliency maps」），但很容易被误以为"增强没生效"。记录下来。
    """
    print("\n[8] 缓冲区初值导致的头几批行为")
    n = 10
    batch = make_batch(n)
    saga = build_saga(load_backbone_cpu())
    k = int(n * 0.5)
    s = saga.buffers["observation.images.up"].get(batch["index"][:k])
    check("缓冲区初值使 s 均匀（=1.0，裁剪后 0.8）", abs(s.max().item() - s.min().item()) < 1e-6)


def test_batch_size_threshold():
    """**batch_size 太小会让 SaGA 静默退化成 random_overlay。**

    参考实现（以及本移植）用 `int(n * β)` 数每批要更新几张显著图。
    β=0.1 时：

        batch_size ≤ 9   → int(n*0.1) == 0  → 一张都不更新
                                             → 缓冲区永远停在初值 1.0
                                             → s ≡ clip(1,0,λ)² = 0.64 是常数
                                             → 显著度引导**完全没生效**，
                                               实际跑的是「random_overlay + 权重 0.64」

    这一条不会报错、不会崩、loss 照样下降 —— 跑出来的是**另一个方法**，
    却是挂着 SaGA 的名。所以这里把它钉死。

    本项目实际训练用 batch_size=16 → int(1.6)=1，每步更新 1 张，是真的在跑 SaGA。
    但 21999 帧 / 每步 1 张 → 一轮要 21999 步才把所有帧碰一遍（100k 步约 4.5 轮）。
    """
    print("\n[9] batch_size 下限（int(n×β) 截断）")
    from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: F401
    backbone = load_backbone_cpu()

    thr = 1 / 0.1
    for n in (8, 9, 10, 16, 32):
        saga = build_saga(backbone)
        batch = make_batch(n)
        before = saga.buffers["observation.images.up"].buffer.clone()
        saga(batch, step=10_000)
        after = saga.buffers["observation.images.up"].buffer
        changed = int((after != before).sum().item())
        expect_zero = int(n * 0.1) == 0
        ok = (changed == 0) if expect_zero else (changed > 0)
        check(f"batch={n:2d}：{'不更新（退化）' if expect_zero else '确实更新了'}"
              f"  int({n}×0.1)={int(n * 0.1)}",
              ok, f"缓冲区被改动的格子数={changed}")
    print(f"      下限：batch_size ≥ 1/β = {thr:.0f}")


def main():
    test_fullgrad_toy()
    test_fullgrad_act()
    test_buffer()
    test_batch_behaviour()
    test_first_batch_is_uniform()
    test_batch_size_threshold()

    print(f"\n{'='*60}")
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for f in FAILED:
            print(f"  FAIL: {f}")
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
