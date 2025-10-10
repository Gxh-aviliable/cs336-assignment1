import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device:torch.device|None=None):
        super().__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        self.device=device

        self.cos,self.sin=self.precompute_rotary_emb(d_k,theta,max_seq_len)
    def precompute_rotary_emb(self,d_k,theta:float,max_seq_len):
        k=torch.arange(d_k/2,device=self.device) #(dim/2,)
        i = torch.arange(max_seq_len, device=self.device)  # (T,)
        thetas=1.0/(theta**(2*k/ d_k))
        i=i.unsqueeze(dim=1)
        thetas=thetas.unsqueeze(dim=0)
        angles=i*thetas
        cos=torch.cos(angles)# T,d/2
        sin=torch.sin(angles) # T,d/2
        return cos ,sin
