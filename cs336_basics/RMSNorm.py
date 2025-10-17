import torch
import torch.nn as nn
import math


class RMSNorm(nn.Module):
    def __init__( self,
    d_model: int,
    eps: float = 1e-5,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        factory_kwargs={"device":device,"dtype":dtype}
        self.weight=nn.Parameter(torch.ones(self.d_model,**factory_kwargs))
    def forward(self,x:torch.Tensor):
        in_dtype=x.dtype
        x=x.to(torch.float32)
        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)
        y=inv_rms *x #点乘
        y=y*self.weight.to(torch.float32)
        return y.to(in_dtype)




