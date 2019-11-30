# %matplotlib inline
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()
    print("rand:",torch.rand(X.shape))
    print('mask：',mask)
    return mask * X / keep_prob

X = torch.arange(10).view(2, 5)
print(X)

print("0",dropout(X, 0))
print("0.5",dropout(X, 0.5))
print("0.9",dropout(X, 0.9))
print("1",dropout(X, 1))


