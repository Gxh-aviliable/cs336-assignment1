import torch
import torch.nn as nn
from .Linear import Linear

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model :int ,d_ff:int ,device:torch.device | None=None ,dtype:torch.dtype | None=None):
        super().__init__()
        self.d_model=d_model
        self.d_ff=d_ff
        self.w1=Linear(d_model,d_ff)
        self.w2=Linear(d_ff,d_model)
        self.w3=Linear(d_model,d_ff)
    def silu(self,x:torch.Tensor):
        return x*torch.sigmoid(x)
    def forward(self,x:torch.Tensor):
        return self.w2(self.silu(self.w1(x))*self.w3(x))

