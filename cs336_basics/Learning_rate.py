import torch
import torch.nn
import math
from typing import Iterable
def lr_cosine_schedule(t:int,a_max:float,a_min:float,T_w:int ,T_c:int):
    if t <T_w :
        if T_w <= 0:  # 防止除0 就是不需要预热的时候
            return float(a_max)
        return t/float(T_w) *a_max
    if t<=T_c:
        if T_c == T_w:  # 防止除0
            return float(a_min)
        output=a_min+0.5*(1.0+math.cos(((t-T_w)/float(T_c-T_w))*math.pi))*(a_max-a_min)
        return output
    return a_min

def clip_grad_l2_(parameters: Iterable[torch.nn.Parameter],
                  max_norm: float,
                  eps: float = 1e-6) -> float:
    """
    全局 L2 范数梯度裁剪（in-place）。
    - parameters: 可迭代的 nn.Parameter（如 model.parameters()）
    - max_norm: 允许的梯度 L2 范数上限 M
    - eps: 数值稳定项（默认 1e-6）
    返回：裁剪前的全局梯度 L2 范数（float）
    """
    # 过滤掉没有梯度的参数
    params = [p for p in parameters if p is not None and p.grad is not None]
    if len(params) == 0:
        return 0.0

    if max_norm <= 0:
        # 非法/极端情况：直接清零梯度
        for p in params:
            p.grad.detach_()
            p.grad.zero_()
        return 0.0

    # 计算所有参数梯度的全局 L2 范数
    # 注意对 grad 使用 detach，以免参与 autograd
    device = params[0].grad.device
    total_sq = torch.zeros((), device=device)
    for p in params:
        g = p.grad.detach()
        total_sq += g.pow(2).sum()
    total_norm = total_sq.sqrt()  # Tensor 标量

    # 若超过上限，则按比例缩放： g <- g * (M / (||g||_2 + eps))
    if total_norm > max_norm:
        scale = (max_norm / (total_norm + eps)).to(device)
        for p in params:
            p.grad.mul_(scale)

    return float(total_norm.item())