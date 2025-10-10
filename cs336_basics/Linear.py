import torch
import torch.nn as nn
import math
from einops import rearrange,einsum
class Linear(nn.Module):
    def __init__(self,in_features: int ,out_features :int ,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features =in_features
        self.out_features=out_features
        self.W=self.init_weight(out_features,in_features,factory_kwargs)
        self.weight=nn.Parameter(self.W)

    def init_weight (self,out_features,in_features,factory_kwargs):
        W=torch.empty(out_features,in_features,**factory_kwargs)
        mean=0
        std=math.sqrt(2/(in_features +out_features))
        nn.init.trunc_normal_(W,mean,std,-2*std,2*std)
        return W
    def forward (self,x):
        return x @ self.weight.T

