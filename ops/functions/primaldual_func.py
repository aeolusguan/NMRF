import torch
import NeSyStereo as NeSy


class _pd_tgv_solver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, t, u, u_, v, v_, p, q, lam, huber_delta, alpha_2, tau, sigma, theta):
        u_t_plus_1, u__t_plus_1, v_t_plus_1, v__t_plus_1, p_t_plus_1, p_unscaled, q_t_plus_1, q_unscaled = NeSy.pd_tgv_forward(
            u, u_, v, v_, p, q, f, t, lam, alpha_2, tau, sigma, theta, huber_delta)
        ctx.tau = tau
        ctx.sigma = sigma
        ctx.alpha_2 = alpha_2
        ctx.huber_delta = huber_delta
        ctx.theta = theta
        ctx.save_for_backward(t, f, u_, v_, u_t_plus_1, p_unscaled, p_t_plus_1, q_unscaled, lam)

        return u_t_plus_1, u__t_plus_1, v_t_plus_1, v__t_plus_1, p_t_plus_1, q_t_plus_1

    @staticmethod
    def backward(ctx, grad_u, grad_u_, grad_v, grad_v_, grad_p, grad_q):
        t, f, u__in, v__in, u_t_plus_1, p_unscaled, p_t_plus_1, q_unscaled, lam = ctx.saved_tensors
        grad_u = grad_u.contiguous()
        grad_u_ = grad_u_.contiguous()
        grad_v = grad_v.contiguous()
        grad_v_ = grad_v_.contiguous()
        grad_p = grad_p.contiguous()
        grad_q = grad_q.contiguous()

        alpha_2 = ctx.alpha_2
        tau = ctx.tau
        sigma = ctx.sigma
        theta = ctx.theta
        huber_delta = ctx.huber_delta
        grad_u_in, grad_u__in, grad_v_in, grad_v__in, grad_p_in, grad_q_in, grad_t, grad_f, grad_lam = \
            NeSy.pd_tgv_backward(t, f, u__in, v__in, u_t_plus_1, p_unscaled, p_t_plus_1, q_unscaled, grad_u, grad_u_,
                                 grad_v, grad_v_, grad_p, grad_q, lam, alpha_2, tau, sigma,
                                 theta, huber_delta)
        return grad_f, grad_t, grad_u_in, grad_u__in, grad_v_in, grad_v__in, grad_p_in, grad_q_in, grad_lam, None, None, None, None, None


pd_tgv_solver = _pd_tgv_solver.apply
