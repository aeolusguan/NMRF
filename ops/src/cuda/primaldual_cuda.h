#pragma once

#include <torch/extension.h>
#include <vector>


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> Kop_forward_cuda(const torch::Tensor &t, const torch::Tensor &u, const torch::Tensor &v);

std::vector<torch::Tensor> Kdiv_forward_cuda(const torch::Tensor &t, const torch::Tensor &p, const torch::Tensor &q);

std::vector<torch::Tensor> pd_tgv_forward_cuda(
    const torch::Tensor &u,
    const torch::Tensor &u_,
    const torch::Tensor &v,
    const torch::Tensor &v_,
    const torch::Tensor &p,
    const torch::Tensor &q,
    const torch::Tensor &f,
    const torch::Tensor &t,
    const torch::Tensor &lambda,
    double alpha_2,
    double tau,
    double sigma,
    double theta,
    double huber_delta);

std::vector<torch::Tensor> pd_tgv_backward_cuda(
    const torch::Tensor &t,
    const torch::Tensor &f,
    const torch::Tensor &u__in,
    const torch::Tensor &v__in,
    const torch::Tensor &u,
    const torch::Tensor &p_unscaled,
    const torch::Tensor &p,
    const torch::Tensor &q_unscaled,
    const torch::Tensor &grad_u,
    const torch::Tensor &grad_u_,
    const torch::Tensor &grad_v,
    const torch::Tensor &grad_v_,
    torch::Tensor grad_p,
    torch::Tensor grad_q,
    const torch::Tensor &lambda,
    double alpha_2,
    double tau,
    double sigma,
    double theta,
    double huber_delta);