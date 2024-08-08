import torch
import math
import NeSyStereo as NeSy
import unittest
from functions.primaldual_func import pd_tgv_solver


def _gradient(u):
    dx = torch.zeros_like(u)
    dy = torch.zeros_like(u)
    dx[..., :-1] = u[..., 1:] - u[..., :-1]
    dy[..., :-1, :] = u[..., 1:, :] - u[..., :-1, :]
    return dx, dy


def _build_diffusion_tensor(I):
    """
    I (Tensor): gray image [B,H,W]

    Returns:
        diffusion tensor: [B,H,W,2,2] exp(-\gamma|\nabla I|^\beta)nn^T + n_p.n_p^T
    """
    beta, gamma = 1.0, 4.0
    dx, dy = _gradient(I)
    norm = torch.sqrt(torch.square(dx) + torch.square(dy))
    nx = dx / torch.clamp(norm, min=1e-4)
    ny = dy / torch.clamp(norm, min=1e-4)
    n = torch.stack((ny, nx), dim=-1).unsqueeze(-1)
    n_p = torch.stack((-nx, ny), dim=-1).unsqueeze(-1)
    t = torch.exp(-gamma * torch.pow(norm, beta))[..., None, None] * n @ n.transpose(-2, -1) + n_p @ n_p.transpose(-2, -1)
    return t


class Test_PrimaldualToolbox(unittest.TestCase):
    def setUp(self):
        self.B = 20
        self.H = 55
        self.W = 55
        self.I = torch.rand(self.B, self.H, self.W).cuda()

    def test_operator(self, dtype=torch.float32):
        I = self.I.to(dtype)
        T = _build_diffusion_tensor(I)

        u = torch.rand(self.B, self.H, self.W, dtype=dtype).cuda()
        vx, vy = _gradient(u)
        v = torch.stack((vy, vx), dim=-1)
        p = torch.randn(self.B, self.H, self.W, 2).cuda()
        q = torch.randn(self.B, self.H, self.W, 3).cuda()
        uop, vop = NeSy.Kop_forward(T, u, v)
        udiv, vdiv = NeSy.Kdiv_forward(T, p, q)

        # <y, Kx>
        yKx = torch.sum(p * uop, dim=-1, keepdim=False) + q[..., 0] * vop[..., 0] + 2 * q[..., 1] * vop[..., 1] + q[..., 2] * vop[..., 2]
        yKx = torch.sum(torch.sum(yKx, dim=-1, keepdim=False), dim=-1, keepdim=False)

        # <x, K*y>
        xKy = u * udiv + torch.sum(v * vdiv, dim=-1, keepdim=False)
        xKy = torch.sum(torch.sum(xKy, dim=-1, keepdim=False), dim=-1, keepdim=False)

        self.assertTrue(torch.allclose(yKx, -xKy))

    def test_tgv_forward(self, dtype=torch.float32):
        I = self.I.to(dtype)
        T = _build_diffusion_tensor(I)

        x = torch.arange(0, self.W, dtype=dtype).view(1, self.W).repeat(self.H, 1)
        y = torch.arange(0, self.H, dtype=dtype).view(self.H, 1).repeat(1, self.W)
        f = (2 * x + 3 * y).cuda().to(dtype)
        f = f[None] + torch.randn(self.B, self.H, self.W).cuda().to(dtype)
        lam = torch.rand(self.B, self.H, self.W).cuda().to(dtype)
        huber_delta = 0.5
        alpha_2 = 0.1
        tau = sigma = 1.0 / math.sqrt(12.0)
        theta = 1.0
        u = f.clone()
        u_ = f.clone()
        dx, dy = _gradient(f)
        v = torch.stack((dy, dx), dim=-1)
        v_ = v.clone()
        p = torch.zeros_like(v)
        q = torch.zeros(f.size() + (3,), dtype=dtype, device=f.device)

        for _ in range(20000):
            u_t_plus_1, u__t_plus_1, v_t_plus_1, v__t_plus_1, p_t_plus_1, q_t_plus_1 = pd_tgv_solver(f, T, u, u_, v,
                                                                                                     v_, p, q, lam,
                                                                                                     huber_delta,
                                                                                                     alpha_2, tau,
                                                                                                     sigma, theta)
            if torch.allclose(u_t_plus_1, u, rtol=0, atol=1e-5):
                break
            u = u_t_plus_1
            u_ = u__t_plus_1
            v = v_t_plus_1
            v_ = v__t_plus_1
            p = p_t_plus_1
            q = q_t_plus_1

        u_opt, v_opt = u, v

        # check the optimal condition: first order derivative equals to zero
        g_u = lam * (u_opt - f)
        Ku, Kv = NeSy.Kop_forward(T, u_opt, v_opt)
        f_Ku = torch.zeros_like(Ku)
        mask = (Ku < huber_delta) & (Ku > -huber_delta)
        f_Ku[mask] = Ku[mask] / huber_delta
        f_Ku[Ku <= -huber_delta] = -1
        f_Ku[Ku >= huber_delta] = 1

        f_Kv = torch.zeros_like(Kv)
        mask = Kv.abs() < 1e-4
        Kv[mask] = 0
        Kv_norm = torch.sqrt(torch.square(Kv[..., 0]) + 2 * torch.square(Kv[..., 1]) + torch.square(Kv[..., 2]))
        mask = Kv[..., 0] != 0
        f_Kv[..., 0][mask] = alpha_2 * Kv[..., 0][mask] / Kv_norm[mask]
        mask = Kv[..., 1] != 0
        f_Kv[..., 1][mask] = alpha_2 * Kv[..., 1][mask] / Kv_norm[mask]
        mask = Kv[..., 2] != 0
        f_Kv[..., 2][mask] = alpha_2 * Kv[..., 2][mask] / Kv_norm[mask]
        Kdiv_u, Kdiv_v = NeSy.Kdiv_forward(T, f_Ku, f_Kv)

        du = g_u - Kdiv_u
        dv = -Kdiv_v
        self.assertTrue(torch.all(du.abs() < 1e-3).cpu().item())
        # we do not check the gradient w.r.t v, since its approximation to
        # non-differentiable
        # self.assertTrue(torch.all(dv.abs() < 1e-3).cpu().item())

    def test_tgv_backward(self):
        self.B = 2
        self.H = 8
        self.W = 8

        I = torch.rand(self.B, self.H, self.W).cuda()
        T = _build_diffusion_tensor(I)
        T = T.to(torch.float64)

        x = torch.arange(0, self.W).view(1, self.W).repeat(self.H, 1)
        y = torch.arange(0, self.H).view(self.H, 1).repeat(1, self.W)
        f = (2 * x + 3 * y).cuda()
        f = f[None] + torch.randn(self.B, self.H, self.W).cuda().to(torch.float64)
        lam = torch.rand(self.B, self.H, self.W).cuda().to(torch.float64)
        huber_delta = 0.5
        alpha_2 = 0.1
        tau = sigma = 1.0 / math.sqrt(12.0)
        theta = 1.0
        u = f.clone()
        u_ = f.clone()
        dx, dy = _gradient(f)
        v = torch.stack((dy, dx), dim=-1)
        v_ = v.clone()
        p = torch.zeros_like(v)
        q = torch.zeros(f.size() + (3,), dtype=f.dtype, device=f.device)
        f.requires_grad_()
        T.requires_grad_()
        u.requires_grad_()
        u_.requires_grad_()
        v.requires_grad_()
        v_.requires_grad_()
        p.requires_grad_()
        q.requires_grad_()
        lam.requires_grad_()
        torch.autograd.gradcheck(pd_tgv_solver,
                                 (f, T, u, u_, v, v_, p, q, lam, huber_delta, alpha_2, tau, sigma, theta),
                                 eps=1e-6, atol=1e-3, nondet_tol=1e-4)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main(verbosity=2)