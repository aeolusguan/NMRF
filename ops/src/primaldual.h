#pragma once

#include "cuda/primaldual_cuda.h"

std::vector<torch::Tensor> Kop_forward(const torch::Tensor &t,
                                       const torch::Tensor &u,
                                       const torch::Tensor &v)
{
    CHECK_INPUT(t);
    CHECK_INPUT(u);
    CHECK_INPUT(v);
    return Kop_forward_cuda(t, u, v);
}

std::vector<torch::Tensor> Kdiv_forward(const torch::Tensor &t,
                                     const torch::Tensor &p,
                                     const torch::Tensor &q)
{
    CHECK_INPUT(t);
    CHECK_INPUT(p);
    CHECK_INPUT(q);
    return Kdiv_forward_cuda(t, p, q);
}

std::vector<torch::Tensor> pd_tgv_forward(
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
    double huber_delta)
{
    CHECK_INPUT(u);
    CHECK_INPUT(u_);
    CHECK_INPUT(v);
    CHECK_INPUT(v_);
    CHECK_INPUT(p);
    CHECK_INPUT(q);
    CHECK_INPUT(f);
    CHECK_INPUT(t);
    CHECK_INPUT(lambda);
    return pd_tgv_forward_cuda(u, u_, v, v_, p, q, f, t, lambda, alpha_2, tau, sigma, theta, huber_delta);
}

std::vector<torch::Tensor> pd_tgv_backward(
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
    double huber_delta)
{
    CHECK_INPUT(t);
    CHECK_INPUT(f);
    CHECK_INPUT(u__in);
    CHECK_INPUT(v__in);
    CHECK_INPUT(u);
    CHECK_INPUT(p_unscaled);
    CHECK_INPUT(p);
    CHECK_INPUT(q_unscaled);
    CHECK_INPUT(grad_u);
    CHECK_INPUT(grad_u_);
    CHECK_INPUT(grad_v);
    CHECK_INPUT(grad_v_);
    CHECK_INPUT(grad_p);
    CHECK_INPUT(grad_q);
    CHECK_INPUT(lambda);
    return pd_tgv_backward_cuda(t, f, u__in, v__in, u, p_unscaled, p, q_unscaled, grad_u, grad_u_, grad_v, grad_v_, grad_p, grad_q, lambda, alpha_2, tau, sigma, theta, huber_delta);
}