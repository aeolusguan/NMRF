#pragma once

#include <torch/extension.h>

at::Tensor downsample_forward_cuda(const at::Tensor& input,
                                   const at::Tensor& label,
                                   const float nms_thresh);
