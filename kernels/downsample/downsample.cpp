#include <torch/torch.h>
#include <torch/extension.h>

at::Tensor downsample_forward_cuda(const at::Tensor& input,
                                   const at::Tensor& label,
                                   const float nms_thresh);

// Interface for Python
at::Tensor downsample_forward(const at::Tensor& input,
                              const at::Tensor& label,
                              const float nms_thresh) {
  if (input.is_cuda()) {
    return downsample_forward_cuda(input, label, nms_thresh);
  }
  AT_ERROR("Not implemented on the CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("downsample_forward", &downsample_forward, "forward pass for downsample operator");
}