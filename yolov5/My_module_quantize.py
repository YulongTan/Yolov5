import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np

torch.set_default_dtype(torch.float64)

def quantize(x, num_bits, fl_bits):
    FL = fl_bits
    IL = num_bits - fl_bits
    # MIN = -(1 << (IL - 1))
    MIN = -2 ** (IL - 1)
    MAX = -MIN - 2 ** (-FL)
    q = torch.floor((x * (2 ** FL))) / 2 ** FL   # torch.floor
    q_copy = q
    # q = torch.clip(q, MIN, MAX)
    q = torch.clamp(q, MIN, MAX)
    # print(torch.max(torch.abs(q_copy - q)))
    return q
