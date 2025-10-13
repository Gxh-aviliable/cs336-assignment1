import torch
import torch.nn as nn
import numpy as np

def get_batch(x:np.ndarray,batch_size:int,context_length,device:torch.device | None=None):
    """
        从一条长序列里随机抽取 batch。
        输入:
            x:            一维 numpy int 数组( token IDs )
            batch_size:   批大小 B
            context_length: 序列长度 m
            device:       'cpu' | 'cuda:0' | 'mps'
            rng:          可选的 numpy 随机数发生器, 便于可复现
        输出:
            (inputs, targets): 两个 long tensor, 形状都是 (B, m)，放在指定 device 上
                               targets 是 inputs 在时间上右移一位（next-token）
        """
    assert x.ndim==1 ,"x 是一个一维数组" #一维数组
    n=x.shape[0]
    need=context_length+1
    assert need<n ,"序列的长度太短了"
    rng = np.random.default_rng()
    # 选择设备并做可用性检查
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA 设备，但当前不可用")
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("指定了 MPS 设备，但当前不可用")

    windows=np.lib.stride_tricks.sliding_window_view(x,need) #生成滑动窗口 生成的是2维度数组
    max_start=n-need+1
    starts=rng.integers(0,max_start,size=batch_size,endpoint=False) #starts 生成 batch_size 个 [0, max_start) 的随机整数(batch_size,)
    batch=windows[starts]# numpy 允许直接把列表当作索引传入
    batch_inputs=batch[:,:context_length]
    batch_targets=batch[:,1:]


    pin = (dev.type == "cuda")  # 仅在 CUDA 下 pin_memory 有意义
    inputs = torch.as_tensor(batch_inputs, dtype=torch.long, device="cpu").pin_memory() if pin \
        else torch.as_tensor(batch_inputs, dtype=torch.long, device="cpu")
    targets = torch.as_tensor(batch_targets, dtype=torch.long, device="cpu").pin_memory() if pin \
        else torch.as_tensor(batch_targets, dtype=torch.long, device="cpu")

    # 非阻塞搬运（CUDA/MPS 下有效）
    inputs = inputs.to(dev, non_blocking=True)  #把张量从 CPU 拷到 目标设备
    targets = targets.to(dev, non_blocking=True)
    return inputs, targets

    return None
if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5, 6])  # 一维数组
    need = 3  # 窗口大小
