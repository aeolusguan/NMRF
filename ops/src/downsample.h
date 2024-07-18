#pragma once

#ifdef WITH_CUDA
#include "cuda/downsample_cuda.h"
#endif


at::Tensor downsample_forward(
    const at::Tensor& input,
    const at::Tensor& label,
    const float nms_thresh)
{
    if (input.is_cuda())
    {
#ifdef WITH_CUDA
    return downsample_forward_cuda(input, label, nms_thresh);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}