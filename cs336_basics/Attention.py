import torch
import torch.nn as nn
import math
from einops import einsum, rearrange
from .RotaryPositionalEmbedding import RotaryPositionalEmbedding
from .Linear import Linear


def softmax(x:torch.Tensor,dimension:int):
    """x: B S D"""
    x_max =torch.max(x,dim=dimension,keepdim=True).values
    x_stable=x-x_max
    
    x_esp=torch.exp(x_stable)
  
    x_total=torch.sum(x_esp,dim=dimension,keepdim=True)
    return x_esp/x_total  #每个类别的得分



def scaled_dot_product_attention(
        Q:torch.Tensor,
        K:torch.Tensor,
        V:torch.Tensor,
        mask:torch.Tensor| None=None
):
    """Q: B S D"""
    d_k=K.shape[-1]
    #attention_score = einsum(Q,K,"... query d_k, ... keys d_k -> ... query keys")/math.sqrt(d_k)
    attention_score = (Q @ K.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None :
        attention_score=attention_score.masked_fill(mask,float('-inf'))
    output=softmax(attention_score,dimension=-1)@V
    return output


class MultiheadSelfAttention(nn.Module):

    def __init__(self,d_model:int,num_heads:int,use_rope:bool= False,max_seq_len: int | None = None, 
                 theta: float | None = None, token_positions: torch.Tensor | None = None):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.use_rope=use_rope
        self.q_proj=Linear(d_model,d_model)
        self.k_proj=Linear(d_model,d_model)
        self.v_proj=Linear(d_model,d_model)
        self.o_proj=Linear(d_model,d_model)
        self.token_positions=token_positions
        self.rope=RotaryPositionalEmbedding(theta,d_model//num_heads,max_seq_len) if use_rope else None
        #self.register_buffer("causal_upper",torch.triu(torch.ones(max_seq_len,max_seq_len,dtype=torch.bool),diagonal=1),persistent=False)

    def forward(self,in_features:torch.Tensor):
        batch_size,seq_len,_=in_features.shape
        q=self.q_proj (in_features).view(batch_size,seq_len,self.num_heads,self.d_model//self.num_heads).transpose(1,2)
        k=self.k_proj (in_features).view(batch_size,seq_len,self.num_heads,self.d_model//self.num_heads).transpose(1,2)
        v=self.v_proj (in_features).view(batch_size,seq_len,self.num_heads,self.d_model//self.num_heads).transpose(1,2) #现在是 B H S D/H
        device = q.device
        if self.rope :
            q=self.rope(q,self.token_positions)
            k=self.rope(k,self.token_positions)

        causal_mask=torch.triu(torch.ones(seq_len,seq_len,dtype=torch.bool,device=device),diagonal=1)
        attn =scaled_dot_product_attention(q,k,v,causal_mask).transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model) #现在是 Ｂ　Ｈ　Ｓ
        result =self.o_proj(attn)
        return result



if __name__ == "__main__":
    x = torch.tensor([
        [1, 3, 2],
        [4, 1, 5]
    ])  # 形状：(2, 3) (T,d)
    softmax(x,dimension=-1)