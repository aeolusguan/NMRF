import torch
from torch import nn
import math
import downsample_sp



class _DownSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, label, nms_thresh):
        return downsample_sp.downsample_forward(input, label, nms_thresh)
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None

downsample = _DownSample.apply
