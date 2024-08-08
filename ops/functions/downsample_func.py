import torch
import NeSyStereo as NeSy


class _DownSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, label, nms_thresh):
        return NeSy.downsample_forward(input, label, nms_thresh)

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None


downsample = _DownSample.apply
