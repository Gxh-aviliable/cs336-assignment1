import torch
import torch.nn as nn
from .Linear import Linear

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model :int ,d_ff:int ,device:torch.device | None=None ,dtype:torch.dtype | None=None):
        super().__init__()
        self.d_model=d_model
        self.d_ff=d_ff
        self.layer1=Linear(d_model,d_ff)
        self.layer2=Linear(d_ff,d_model)
        self.layer3=Linear(d_model,d_ff)
    def silu(self,x:torch.Tensor):
        return x*torch.sigmoid(x)
    def forward(self,x:torch.Tensor):
        return self.layer2(self.silu(self.layer1(x))*self.layer3(x))

