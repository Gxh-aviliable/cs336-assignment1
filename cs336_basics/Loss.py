import torch
import torch.nn as nn
from typing import Callable, Optional, Iterable
import math
def cross_entropy_loss(logits:torch.Tensor,targets=torch.Tensor):
    """logits 的维度 (B,S,Vocab_size)
    targets 的维度 为 B S
    是一个整数，表示第 b 个样本、第 s 个位置的正确单词的词表索引
        tensor([
      [[2.5], [2.0], [0.2]],   # 每行选出对应类别的 logit
      [[0.5], [2.3], [0.8]]
    ])  # (2,3,1)
    """
    lse = torch.logsumexp(logits,dim=-1) # B S
    target_logits=logits.gather(dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)  #取出每个样本在其正确类别上的 logit 值
    correct_loss=lse-target_logits
    loss=torch.mean(correct_loss)
    return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr,betas,eps,weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0:
            raise ValueError(f"Invalid eps: {eps}")
        b1,b2=betas
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    #在这里面我们对p(nn.Parameters()的数据)就是一个带有 requires_grad的参数进行修改,会跟autograd的时候冲突
    """ @torch.no_grad() 是告诉 PyTorch：
“我现在在更新参数，不需要你再跟踪梯度历史。”
否则对参数的原地修改会破坏计算图，从而报错。"""

    @torch.no_grad()
    def step(self,closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr=group["lr"]
            beta1,beta2=group["betas"]
            eps=group["eps"]
            weight_decay=group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state=self.state[p]
                """{
          parameter_1: {'step': 10, 'exp_avg': tensor(...), 'exp_avg_sq': tensor(...)},
          parameter_2: {'step': 3,  'exp_avg': tensor(...), 'exp_avg_sq': tensor(...)},
          ...
        }"""
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)       # 一阶矩估计
                    state["exp_avg_sq"] = torch.zeros_like(p.data)    # 二阶矩估计
                exp_avg=state["exp_avg"]
                exp_avg_sq=state["exp_avg_sq"]
                # t <- t + 1
                state["step"] += 1
                t = state["step"]

                # 一阶/二阶矩更新
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m = β1 m + (1-β1) g
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v = β2 v + (1-β2) g^2

                # 偏置校正后的学习率 alpha_t
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1  # α_t

                # 参数更新（Adam 部分）
                denom = exp_avg_sq.sqrt().add_(eps)  # √v + ε
                p.addcdiv_(exp_avg, denom, value=-step_size)  # θ ← θ - α_t * m / (√v + ε)

                # 解耦权重衰减（与梯度无关，直接对参数衰减）
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)  # θ ← θ - lr*λ*θ

        return loss