import torch
from loguru import logger
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat

from cs336_basics.bpe_tokenizer.tokenizer import BPETokenizer

def nucleus_sampling(probs, top_p):
    # 1) 排序  对每个样本的词表概率按从大到小排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    #计算前缀和 cumsum，第 k 位是前 k 个最大概率的累计和
    #例如：[0.4, 0.35, 0.25] → [0.4, 0.75, 1.0]。
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)


    # 2) 找到“应当删除”的位置：累计和 > p 的地方
    sorted_indices_to_remove = cumulative_probs > top_p #tok_p 假设为0.5
    # tensor([[False, True, True]]) 就类似于第一个不删除，剩下的删除

    # 3) 右移一位，以“保留第一个使累计超过 p 的 token”
    #    例如 cumsum=[0.4, 0.75, 1.0], p=0.5
    #    >p=[False, True, True] → 右移：[False, False, True]
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # 4) 置零并归一化
    sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
    denom = sorted_probs.sum(dim=-1, keepdim=True)
    sorted_probs = sorted_probs / (denom + 1e-12)
    # 5) 映回原顺序
    filtered = torch.zeros_like(probs)
    filtered.scatter_(1, sorted_indices, sorted_probs)
    return filtered
