import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device:torch.device|None=None):
        super().__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        self.device=device

        i=torch.arange(0,d_k//2,device=device)
        inv_freq=theta**(-2*i/d_k)#(d_k//2,)
        self.register_buffer("inv_freq",inv_freq,persistent=False)
        positions = torch.arange(self.max_seq_len,device=device) #(T,)
        freqs=inv_freq.unsqueeze(0)*positions.unsqueeze(1)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        self.register_buffer("cos_cached", cos, persistent=False)  # (max_seq_len, d_k//2)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _select_cos_sin(self,x:torch.Tensor,token_positions:torch.Tensor):
        B,_,S,_=x.shape
        if token_positions is None:
            token_positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
        flat=token_positions.reshape(-1) # 从 batch ，seq 变为 total_tokens [[1,3,5],[2,0,4]] 每个token 对应的位置？
        cos=self.cos_cached.index_select(0,flat).reshape(*token_positions.shape,self.d_k//2)  #index_select(0, flat) 表示沿第 0 维（位置维），根据 flat 中的索引选取对应行。例如，若 flat = [1, 3, 5, 2, 0, 4]，则会选取 cos_cached[1], cos_cached[3], ..., cos_cached[4]
        sin=self.sin_cached.index_select(0,flat).reshape(*token_positions.shape,self.d_k//2) #B S D//2
        """[
    [[cos1_0, cos1_1], [cos3_0, cos3_1], [cos5_0, cos5_1]],  # 第0个样本
    [[cos2_0, cos2_1], [cos0_0, cos0_1], [cos4_0, cos4_1]]   # 第1个样本
]"""
        return cos, sin
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor):

        cos,sin=self._select_cos_sin(x,token_positions)
        x_even=x[...,0::2]
        x_odd=x[...,1::2]
        x_rot_even=cos*x_even-sin*x_odd
        x_rot_odd =cos*x_odd +sin*x_even

        out=torch.empty_like(x)
        out[...,0::2]=x_rot_even
        out[...,1::2]=x_rot_odd
        return out

