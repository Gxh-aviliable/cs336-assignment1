import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
        device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        # (vocab_size, d_model) 参数矩阵
        self.embedding_dim=embedding_dim
        factory_kwargs={"device":device,"dtype":dtype}
        W=torch.empty(self.num_embeddings,self.embedding_dim,**factory_kwargs)
        mean=0
        std=0.02
        W_init = nn.init.trunc_normal_(W,mean,std)
        self.weight=nn.Parameter(W_init)
    def forward(self,token_ids):
        """token_id 是 batch_size seq_len"""
        return self.weight[token_ids.to(self.weight.device)]



