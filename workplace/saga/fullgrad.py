"""FullGrad —— 移植自 RoboSaGA 的 `robosaga/fullgrad.py` + `tensor_extractors.py`。

出处：Zhuang, Wang, Ingelhag, Kyrki, Kragic,
      "Enhancing Visual Domain Robustness in Behaviour Cloning via
       Saliency-Guided Augmentation", CoRL 2024.
原文的 FullGrad 又是改自 idiap/fullgrad-saliency（Srinivas & Fleuret, NeurIPS 2019）。

## 为什么公式一个字都不改

因为我先前自己按论文正文写过一版，**写错了**：我以为是 `grad ⊙ activation`，
实际参考实现是 `grad ⊙ bias`（`fullgrad.py:59` 的 `g * b`）。
这类"看着像、但其实是另一族方法"的错误，靠读正文是发现不了的 ——
所以这一份以**源码为准**，只在接 ACT 的地方动刀，并逐处标注。

## 不能照抄的地方：梯度怎么取（适配 1，也是最要紧的一处）

参考实现用 `module.register_full_backward_hook` 拿「各层输出的梯度」。这条路在
torchvision 的 resnet18 上**有两道坎**，两道都踩过：

**坎一：`register_full_backward_hook` 直接报错。** 它会把模块输出包成自定义
Function 的输出（即输入的 view），而 torchvision 的 residual block 里到处是原地
操作（`nn.ReLU(inplace=True)`、`out += identity`，`resnet.py:83/102/139/160`），
autograd 会报
"Output 0 of BackwardHookFunctionBackward is a view and is being modified inplace"。
关掉 relu 的 inplace 也没用 —— `out += identity` 是同样的坑。
（论文原环境大概率是更早的 torch，那会儿这个检查还没加上。）

**坎二：改用 `autograd.grad(f, 各层输出)` 会静默算错。** 看起来"换个方式取同一个量"，
其实不是：原地 relu 会把被捕获张量的 `grad_fn` 改写成 `ReluBackward`，
于是 autograd 把它当成 relu 的**输出**来求导，返回的是**过完 relu 的**梯度，
而不是该层输出的梯度。实测：

    z = x * 2                       # 假设这是 hook 捕获的某层输出
    f = (z.relu_() * 3).sum()
    autograd.grad(f, z)  ->  [3, 3, 3]    # 错：relu 之后
    应为                  ->  [3, 0, 3]    # 对：relu 之前

这个错**不会报任何异常**，只是让显著图悄悄变样。接上真 ACT 时它表现为完备性差 6%
（`raw=527.96` vs `fullgrad=495.15`）—— 完备性测试的价值正在于此。

**最后用的办法**：前向时用 forward hook 拿到各层输出张量，立刻在它上面
`register_hook`。实测**张量 hook 不受原地操作影响**（上面那个例子里它拿到的仍是
`[3, 0, 3]`），因为它不往计算图里插自定义 Function。
输入梯度仍用一次 `autograd.grad(f, x)` —— 这一次调用同时触发反向、把张量 hook 点亮。

> 顺带记一条：`out += identity` 这种原地**加**是无害的（加法是线性的，
> 加之前和加之后的梯度相等）。真正致命的只有原地**非线性**（relu）。

## 其余四处 ACT 适配

2. `target_modules` 加入 `FrozenBatchNorm2d`。ACT 的 resnet18 用它而不是
   `nn.BatchNorm2d`（`modeling_act.py` 里 `norm_layer=FrozenBatchNorm2d`），
   **不加会被静默跳过** —— resnet18 里那是 20 层，显著图会明显变样。
3. `_effective_bias` 多一个 `FrozenBatchNorm2d` 分支。torchvision 的冻结 BN 前向是
   `(x − rm)/√(rv + eps)·w + b`，展开成 `x·(w/√(rv+eps)) + (b − rm·w/√(rv+eps))`，
   与参考实现对 `nn.BatchNorm2d` 用的 effective bias **是同一个式子**，所以直接照抄。
4. `_raw_output_to_scalar`：ACT 的 backbone 是 `IntermediateLayerGetter`，前向返回
   **dict**（`{"feature_map": (B,512,15,20)}`），取 `["feature_map"].sum()`。
5. 梯度与 bias 的**配对**：参考实现用 `biases[i]` 配 `feature_gradients[L-1-i]`，
   前提是「带 bias 的层全都被执行过」—— **ACT 不满足**：
   `IntermediateLayerGetter` 在 `layer4` 就返回，`fc` 从不执行，
   但 `get_biases()` 遍历 `model.modules()` 时会把它收进去。
   两份列表长度不等 → 整条配对错位，而且是**静默**的。
   这里改成「谁前向跑了就记谁」，天然只含真正执行的层，配对不可能错。
   参考实现的 `get_biases()` 因此成了死代码，已删。

## 自带的完备性测试

`checkCompleteness` 是原作者写的，保留。它验证

    Σ(input_grad · x) + Σ(bias_grads) == model(x).sum()

这是移植是否忠实的**最强单项检查**，且不依赖任何外部仓库。
（为什么参考实现里空间可变的 `g ⊙ b` 求和后等于真正的 bias 项：
`Σ_{h,w} (∂f/∂z_{c,h,w})·b_c = b_c · ∂f/∂b_c`，因为 `∂f/∂b_c = Σ_{h,w} ∂f/∂z_{c,h,w}`。）
"""

from __future__ import annotations

from math import isclose

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops.misc import FrozenBatchNorm2d
except ImportError:  # torchvision 版本差异时的兜底
    FrozenBatchNorm2d = None


def _target_modules() -> list[type]:
    mods = [nn.Conv2d, nn.Linear, nn.BatchNorm2d]
    if FrozenBatchNorm2d is not None:
        mods.append(FrozenBatchNorm2d)
    return mods


class FullGradExtractorBase:
    """对应参考实现的 `FullGradExtractorBase`（`tensor_extractors.py:7`）。"""

    def __init__(self, model: nn.Module):
        self.model = model
        # 反向期由张量 hook 填进来：(模块, 该模块输出的梯度)。见文件头适配 1、5。
        self.layer_grads: list[tuple[nn.Module, torch.Tensor]] = []
        self.forward_handles: list = []
        self.target_modules = _target_modules()

    def unregister_hooks(self):
        for handle in self.forward_handles:
            handle.remove()
        self.forward_handles = []

    def reset(self):
        """前向**之前**调用，清掉上一批的引用。"""
        self.layer_grads = []

    def _capture_output(self, m, inputs, output):
        # 与参考实现同一条过滤：只收「有 bias 的层」。
        # 差别：不靠位置配对，谁跑了就记谁 —— fc 这种没执行的层根本进不来。
        if getattr(m, "bias", None) is None:
            return
        if not isinstance(output, torch.Tensor):
            return
        # 只挂 hook，不留张量引用（见文件头适配 1）。
        output.register_hook(self._make_grad_recorder(m))

    def _make_grad_recorder(self, m):
        def record(grad):
            self.layer_grads.append((m, grad))
        return record

    def _register_module_hooks(self, m):
        if not any(isinstance(m, M) for M in self.target_modules):
            return
        self.forward_handles.append(m.register_forward_hook(self._capture_output))

    def _effective_bias(self, m):
        if not any(isinstance(m, M) for M in self.target_modules):
            return None
        # --- 适配 3：FrozenBatchNorm2d ---
        # torchvision 的冻结 BN 不做 running stats 更新，前向即
        #     (x - rm) / sqrt(rv+eps) * w + b  =  x*(w/sqrt(rv+eps)) + (b - rm*w/sqrt(rv+eps))
        # 与参考实现对 BatchNorm2d 写的 effective bias 完全同形。
        if FrozenBatchNorm2d is not None and isinstance(m, FrozenBatchNorm2d):
            return (
                -(m.running_mean * m.weight / torch.sqrt(m.running_var + m.eps)) + m.bias
            ).data
        if isinstance(m, nn.BatchNorm2d):
            return (
                -(m.running_mean * m.weight / torch.sqrt(m.running_var + m.eps)) + m.bias
            ).data
        if m.bias is None:
            return None
        return m.bias.data

    def get_feature_grads(self, x, output_scalar, retain_graph=False):
        """取输入梯度；各层输出梯度由前向期挂上的张量 hook 在本次反向中填好。

        返回 `(input_gradient, [(模块, 该层输出梯度), ...])`。
        """
        # 这一次 autograd.grad 同时触发整条反向，把各层的张量 hook 点亮。
        input_gradients = torch.autograd.grad(
            outputs=output_scalar,
            inputs=x,
            retain_graph=retain_graph,
            allow_unused=True,
        )[0]
        return input_gradients, list(self.layer_grads)

    def register_hooks(self):
        raise NotImplementedError


class EncoderOnly(FullGradExtractorBase):
    """对应参考实现的 `EncoderOnly`：只挂在视觉编码器上。

    论文要的就是「在视觉编码器层级取显著度」，所以 ACT 传 `policy.model.backbone`。
    """

    def __init__(self, model: nn.Module, obs_key: str | None = None):
        super().__init__(model)
        self.register_hooks()

    def register_hooks(self):
        for m in self.model.modules():
            self._register_module_hooks(m)


class FullGrad:
    """对应参考实现的 `robosaga/fullgrad.py: FullGrad`。公式原样。"""

    def __init__(self, model: nn.Module, obs_key: str | None = None,
                 Extractor: type = EncoderOnly):
        self.model = model
        self.model_ext = Extractor(model, obs_key)

    def unregister_hooks(self):
        self.model_ext.unregister_hooks()

    def register_hooks(self):
        if len(self.model_ext.forward_handles) == 0:
            self.model_ext.register_hooks()

    def fullGradientDecompose(self, x, net_input=None):
        self.model.eval()
        if net_input is None:
            net_input = x

        # 注意：requires_grad_() 是**原地**的。调用方必须传一个拷贝，
        # 否则会把训练 batch 的张量也标上 requires_grad。
        x = x.requires_grad_()

        self.model_ext.reset()          # 见文件头适配 1：前向期靠 forward hook 捕获
        out = self.model(net_input)
        output_scalar = self._raw_output_to_scalar(out)
        input_gradient, layer_grads = self.model_ext.get_feature_grads(
            x, output_scalar
        )

        bias_times_gradients = []
        for m, g in layer_grads:
            b = self.model_ext._effective_bias(m)
            if b is None:
                continue
            if len(g.size()) != len(b.size()):
                bias_size = [1] * len(g.size())
                bias_size[1] = b.size(0)
                b = b.view(tuple(bias_size))
            bias_times_gradients.append(g * b.detach())

        return input_gradient, bias_times_gradients

    def __call__(self, image, net_input=None):
        self.model.eval()
        input_grad, bias_grad = self.fullGradientDecompose(image, net_input)

        with torch.no_grad():
            cam = self._postProcess(input_grad * image).sum(1, keepdim=True)

            im_size = image.size()
            H, W = im_size[2], im_size[3]
            for i in range(len(bias_grad)):
                # 只保留与输入同维（4D）的项 —— 滤掉 Linear 那种 2D 的
                if len(bias_grad[i].size()) == len(im_size):
                    # 与 idiap 原版不同：先按通道求和再插值，省算力（作者注释）
                    temp = self._postProcess(bias_grad[i]).sum(1, keepdim=True)
                    gradient = F.interpolate(
                        temp, size=(H, W), mode="bilinear", align_corners=True
                    )
                    cam += gradient.sum(1, keepdim=True)
        return cam

    def _postProcess(self, layer_grad, eps=1e-6):
        layer_grad = abs(layer_grad)
        flatin = layer_grad.reshape((layer_grad.size(0), -1))
        temp, _ = flatin.min(1, keepdim=True)
        layer_grad = layer_grad - temp.unsqueeze(1).unsqueeze(1)
        flatin = layer_grad.reshape((layer_grad.size(0), -1))
        temp, _ = flatin.max(1, keepdim=True)
        layer_grad = layer_grad / (temp.unsqueeze(1).unsqueeze(1) + eps)
        return layer_grad

    def _raw_output_to_scalar(self, raw_output):
        # --- 适配 4：ACT 的 backbone 是 IntermediateLayerGetter，返回 dict ---
        if isinstance(raw_output, dict):
            if "feature_map" in raw_output:
                return raw_output["feature_map"].sum()
            return sum(v.sum() for v in raw_output.values() if torch.is_tensor(v))
        return raw_output.sum()

    def checkCompleteness(self, x):
        """作者自带的完备性测试，保留原样。移植是否忠实看这一条。"""
        self.model.eval()
        raw_output = self._raw_output_to_scalar(self.model(x))

        self.model_ext.reset()
        input_grad, bias_grad = self.fullGradientDecompose(x)

        fullgradient_sum = (input_grad * x).sum()
        for i in range(len(bias_grad)):
            fullgradient_sum += bias_grad[i].sum()

        err_string = (
            "Completeness test failed! Raw output = "
            + str(raw_output.max().item())
            + " Full-gradient sum = "
            + str(fullgradient_sum.item())
        )
        assert isclose(raw_output.item(), fullgradient_sum.item(), rel_tol=1e-4), (
            err_string + "\nThis is due to incorrect computation of bias-gradients."
        )
        print("Completeness test passed for FullGrad.")
