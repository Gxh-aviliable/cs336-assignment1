import torch
import torch.nn as nn
from.RMSNorm import RMSNorm
from .Attention import MultiheadSelfAttention
from .PositionwiseFeedForward import PositionwiseFeedForward 
from .Embedding import Embedding
from .Linear import Linear



class Transformer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,max_seq_len,theta):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.rms_norm1=RMSNorm(d_model)
        self.rms_norm2=RMSNorm(d_model)

        self.attn=MultiheadSelfAttention(d_model,num_heads,use_rope=True,max_seq_len=max_seq_len,
                                         theta=theta)#token_positions=token_positions
        self.ff=PositionwiseFeedForward(d_model=d_model,d_ff=d_ff)

    def forward(self,in_features:torch.Tensor):
        y=in_features+self.attn(self.rms_norm1(in_features))
        output=y+self.ff(self.rms_norm2(y))
        return output


class TransformerLM(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,rope_theta,
                 vocab_size,context_length,num_layers):
        super().__init__()
        self.vocab_size =vocab_size
        self.context_length=context_length
        self.num_layers=num_layers
        self.token_embeddings=Embedding(num_embeddings=vocab_size,embedding_dim=d_model)
        self.layers=nn.ModuleList([Transformer(d_model=d_model,num_heads=num_heads,d_ff=d_ff,
                                             max_seq_len=self.context_length,
                                             theta=rope_theta) for _ in range(num_layers)])
        self.rms_norm = RMSNorm(d_model=d_model)
        self.output_embeddings = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, in_indices: torch.Tensor):
        """
        Args:
            in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. 
            Shape is (batch_size, sequence_length), where `sequence_length` is at most `context_length`.

        Returns:
            Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
            next-word distribution for each token.
        """
        x = self.token_embeddings(in_indices)

        for layer in self.layers:
            x = layer(x)
        
        x_norm = self.rms_norm(x)
        output_embed = self.output_embeddings(x_norm)
        return output_embed