/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include "ms_deform_attn.h"
#include "downsample.h"
#include "primaldual.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
    m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");
    m.def("downsample_forward", &downsample_forward, "downsample_forward");
    m.def("Kop_forward", &Kop_forward, "linear operator");
    m.def("Kdiv_forward", &Kdiv_forward, "divergence of linear operator");
    m.def("pd_tgv_forward", &pd_tgv_forward, "forward of TGV denoise");
    m.def("pd_tgv_backward", &pd_tgv_backward, "backward of TGV denoise");
}

