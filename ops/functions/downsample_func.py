import torch
import MultiScaleDeformableAttention as MSDA


class _DownSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, label, nms_thresh):
        return MSDA.downsample_forward(input, label, nms_thresh)

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None


downsample = _DownSample.apply
