"""真 RoboSaGA（在线、显著度引导的叠加增强）在 LeRobot ACT 上的实现。

出处：Zhuang, Wang, Ingelhag, Kyrki, Kragic,
"Enhancing Visual Domain Robustness in Behaviour Cloning via Saliency-Guided
Augmentation", CoRL 2024.  论文 PDF 与官方代码在
`workplace/background_generalization/`（`zhuang25b.pdf` / `RoboSaGA-master.zip`）。

对照臂是 `workplace/background_generalization/` 的**离线硬掩码**背景替换 ——
那不是 RoboSaGA，是被刻意保留下来当对照的。
"""

from .backgrounds import BackgroundRandomizer
from .buffer import BufferManager
from .fullgrad import FullGrad, EncoderOnly
from .saga import RoboSaGA, normalize_smaps

__all__ = [
    "BackgroundRandomizer",
    "BufferManager",
    "FullGrad",
    "EncoderOnly",
    "RoboSaGA",
    "normalize_smaps",
]
