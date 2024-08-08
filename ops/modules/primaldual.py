import math
import torch
from torch import nn

from ..functions import pd_tgv_solver


def _gradient(u):
    dx = torch.zeros_like(u)
    dy = torch.zeros_like(u)
    dx[..., :-1] = u[..., 1:] - u[..., :-1]
    dy[..., :-1, :] = u[..., 1:, :] - u[..., :-1, :]
    return dx, dy


def build_diffusion_tensor(I):
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


class PDTGVSolver(nn.Module):
    def __init__(self, alpha1_range, alpha2, huber_delta, theta=1.0):
        super().__init__()
        alpha1_min, alpha1_max = alpha1_range
        self.alpha1_min = alpha1_min
        self.alpha1_max = alpha1_max
        self.alpha2 = alpha2
        self.huber_delta = huber_delta
        self.theta = theta
        self.tau = 1.0 / math.sqrt(12.0)
        self.sigma = self.tau

    def forward(self, steps, f, t, w_and_log_b):
        """
        primal dual solver for the optimization problem:
        u^\star = \arg\min_u 1/2 \lambda (u-f)^2 + \alpha_1 |T(\nabla u - v)| + \alpha_2 |\nabla v|
        """
        u = f.clone().detach()
        u_ = u.clone()
        dx, dy = _gradient(f)
        v = torch.stack((dy, dx), dim=-1).detach()
        v_ = v.clone()
        p = torch.zeros_like(v)
        q = p.new_zeros(u.size() + (3,))

        # log_b: uncertainty
        log_b = w_and_log_b[:, 1].clamp(min=0, max=10).detach()  # prevent gradient backpropagation
        lam = torch.exp(-log_b)  # [B,H,W]

        # rescale anisotropic diffusion tensor
        w = torch.sigmoid(w_and_log_b[:, 0]) * (self.alpha1_max - self.alpha1_min) + self.alpha1_min
        t = t * w[..., None, None]

        alpha2 = self.alpha2
        tau = self.tau
        sigma = self.sigma
        theta = self.theta
        huber_delta = self.huber_delta
        if isinstance(lam, float):
            lam = torch.full_like(f, lam)
        for _ in range(steps):
            u, u_, v, v_, p, q = pd_tgv_solver(f, t, u, u_, v, v_, p, q, lam, huber_delta, alpha2, tau, sigma, theta)
        return u, v
