#include <torch/extension.h>
#include <vector>

#define BLOCK 16

//! divergence operator with anisotropic diffusion tensor T(\nabla u)
template<typename scalar_t>
__forceinline__ __device__ void div_anisotropic(const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> p,
                                                const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> t,
                                                unsigned int b, unsigned int x, unsigned int y,
                                                scalar_t &div_p)
{
    // axis: 0: y, 1: x
    div_p = 0;
    if (y > 0) {
        // T\nabla u(y-1,x)_y & T\nabla u(y-1,x)_x
        div_p += t[b][y-1][x][0][0] * p[b][y-1][x][0] + t[b][y-1][x][1][0] * p[b][y-1][x][1];
    }
    if (y < p.size(1) - 1) {
        // T\nabla u(y,x)_y & T\nabla u(y,x)_x
        div_p -= t[b][y][x][0][0] * p[b][y][x][0] + t[b][y][x][1][0] * p[b][y][x][1];
    }
    if (x < p.size(2) - 1) {
        // T\nabla u(y,x)_y & T\nabla u(y,x)_x
        div_p -= t[b][y][x][0][1] * p[b][y][x][0] + t[b][y][x][1][1] * p[b][y][x][1];
    }
    if (x > 0) {
        // T\nabla u(y,x-1)_y & T\nabla u(y,x-1)_x
        div_p += t[b][y][x-1][0][1] * p[b][y][x-1][0] + t[b][y][x-1][1][1] * p[b][y][x-1][1];
    }
    div_p = -div_p;
}

//! divergence operator with symmetric gradient operator
template <typename scalar_t>
__forceinline__ __device__ void div_sym(const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> q,
                                        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> p,
                                        const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> t,
                                        unsigned int b, unsigned int x, unsigned int y,
                                        scalar_t &div_q_x, scalar_t &div_q_y)
{
    // axis: 0: y, 1: x
    scalar_t div_q_xx_x =
        (x > 0) ?
            ((x < q.size(2) - 1) ?
                q[b][y][x][2] - q[b][y][x-1][2] : -q[b][y][x-1][2]) :
            q[b][y][x][2];
    scalar_t div_q_yy_y =
        (y > 0) ?
            ((y < q.size(1) - 1) ?
                q[b][y][x][0] - q[b][y-1][x][0] : -q[b][y-1][x][0]) :
            q[b][y][x][0];
    scalar_t div_q_xy_x =
        (x > 0) ?
            ((x < q.size(2) - 1) ?
                q[b][y][x][1] - q[b][y][x-1][1] : -q[b][y][x-1][1]) :
            q[b][y][x][1];
    scalar_t div_q_xy_y =
        (y > 0) ?
            ((y < q.size(1) - 1) ?
                q[b][y][x][1] - q[b][y-1][x][1] : -q[b][y-1][x][1]) :
            q[b][y][x][1];
    div_q_x = div_q_xx_x + div_q_xy_y;
    div_q_y = div_q_yy_y + div_q_xy_x;

    if (x < q.size(2) - 1) {
        div_q_x += t[b][y][x][0][1] * p[b][y][x][0] + t[b][y][x][1][1] * p[b][y][x][1];
    }
    if (y < q.size(1) - 1) {
        div_q_y += t[b][y][x][0][0] * p[b][y][x][0] + t[b][y][x][1][0] * p[b][y][x][1];
    }
}

//! gradient operator with anistropic diffusion tensor
template<typename scalar_t>
__forceinline__ __device__ void grad_anisotropic(const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> u,
                                                 const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> v,
                                                 const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> t,
                                                 unsigned int b, unsigned int x, unsigned int y,
                                                 scalar_t &grad_u_x, scalar_t &grad_u_y)
{
    const unsigned int xp = x + (x < u.size(2) - 1);
    const unsigned int yp = y + (y < u.size(1) - 1);

    grad_u_x = u[b][y][xp] - u[b][y][x];
    grad_u_y = u[b][yp][x] - u[b][y][x];
    scalar_t vx = (x < u.size(2) - 1) ? v[b][y][x][1] : 0;
    scalar_t vy = (y < u.size(1) - 1) ? v[b][y][x][0] : 0;
    scalar_t grad_e_x = grad_u_x - vx;
    scalar_t grad_e_y = grad_u_y - vy;
    grad_u_x = t[b][y][x][1][1] * grad_e_x + t[b][y][x][1][0] * grad_e_y;
    grad_u_y = t[b][y][x][0][1] * grad_e_x + t[b][y][x][0][0] * grad_e_y;
}

//! symmetric gradient operator for vector-valued input
template<typename scalar_t>
__forceinline__ __device__ void grad_sym(const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> v,
                                         unsigned int b, unsigned int x, unsigned int y,
                                         scalar_t &grad_v_xx, scalar_t &grad_v_yy, scalar_t &grad_v_xy)
{
    const unsigned int xp = x + (x < v.size(2) - 1);
    const unsigned int yp = y + (y < v.size(1) - 1);

    grad_v_xx = v[b][y][xp][1] - v[b][y][x][1];
    grad_v_xy = 0.5 * (v[b][yp][x][1] - v[b][y][x][1] + v[b][y][xp][0] - v[b][y][x][0]);
    grad_v_yy = v[b][yp][x][0] - v[b][y][x][0];
}

//! The linear operator K for regularization term:
// \alpha_1 ||T(\nabla u-v)||_H + \alpha_2 ||\nabla v||_1
// which is formulated as F(Kx), x is the concatenation of u and v
template<typename scalar_t>
__global__ void Kop_kernel(const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> t,
                           const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> u,
                           const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> v,
                           torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> uop,
                           torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vop)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int bs = u.size(0);
    const int ht = u.size(1);
    const int wd = u.size(2);

    if ((x >= wd) || (y >= ht))
        return;

    for (int i = 0; i < bs; ++i) { // i is batch index
        scalar_t grad_u_x, grad_u_y;
        grad_anisotropic(u, v, t, i, x, y, grad_u_x, grad_u_y);
        uop[i][y][x][0] = grad_u_y;
        uop[i][y][x][1] = grad_u_x;

        scalar_t grad_v_xx, grad_v_yy, grad_v_xy;
        grad_sym(v, i, x, y, grad_v_xx, grad_v_yy, grad_v_xy);
        vop[i][y][x][0] = grad_v_yy;
        vop[i][y][x][1] = grad_v_xy;
        vop[i][y][x][2] = grad_v_xx;
    }
}

//! divergence of above linear operator K
template<typename scalar_t>
__global__ void Kdiv_kernel(const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> t,
                            const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> p,
                            const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> q,
                            torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> uop,
                            torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vop)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int bs = p.size(0);
    const int ht = p.size(1);
    const int wd = p.size(2);

    if ((x >= wd) || (y >= ht))
        return;

    for (int i = 0; i < bs; ++i) { // i is batch index
        scalar_t div_u;
        div_anisotropic(p, t, i, x, y, div_u);
        uop[i][y][x] = div_u;

        scalar_t div_v_x, div_v_y;
        div_sym(q, p, t, i, x, y, div_v_x, div_v_y);
        vop[i][y][x][0] = div_v_y;
        vop[i][y][x][1] = div_v_x;
    }
}

std::vector<torch::Tensor> Kop_forward_cuda(const torch::Tensor &t,
                                            const torch::Tensor &u,
                                            const torch::Tensor &v)
{
    const auto bs = u.size(0);
    const auto ht = u.size(1);
    const auto wd = u.size(2);

    const dim3 blocks((wd + BLOCK - 1) / BLOCK,
                      (ht + BLOCK - 1) / BLOCK);

    const dim3 threads(BLOCK, BLOCK);

    auto opts = u.options();
    auto uop = torch::zeros({bs, ht, wd, 2}, opts);
    auto vop = torch::zeros({bs, ht, wd, 3}, opts);

    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "Kop_kernel", ([&] {
        Kop_kernel<scalar_t><<<blocks, threads>>>(
            t.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            v.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            uop.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            vop.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
    }));

    return {uop, vop};
}

std::vector<torch::Tensor> Kdiv_forward_cuda(const torch::Tensor &t,
                                             const torch::Tensor &p,
                                             const torch::Tensor &q)
{
    const auto bs = p.size(0);
    const auto ht = p.size(1);
    const auto wd = p.size(2);

    const dim3 blocks((wd + BLOCK - 1) / BLOCK,
                      (ht + BLOCK - 1) / BLOCK);

    const dim3 threads(BLOCK, BLOCK);

    auto opts = p.options();
    auto div_u = torch::zeros({bs, ht, wd}, opts);
    auto div_v = torch::zeros({bs, ht, wd, 2}, opts);

    AT_DISPATCH_FLOATING_TYPES(t.scalar_type(), "Kdiv_kernel", ([&] {
        Kdiv_kernel<scalar_t><<<blocks, threads>>>(
            t.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            p.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            q.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            div_u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            div_v.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
    }));

    return {div_u, div_v};
}

//! ======================= end of linear operator test ===================
//! =======================================================================

//! data term is quadratic \lambda / 2 * ||u-f||^2
template<typename scalar_t>
__global__ void TV_primal_u_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> t,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> f,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> p,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> lambda,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> u,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> u__t_plus_1,
    scalar_t tau,
    scalar_t theta)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int bs = f.size(0);
    const int ht = f.size(1);
    const int wd = f.size(2);

    if ((x >= wd) || (y >= ht))
        return;

    for (int i = 0; i < bs; ++i) { // i is batch index
        // remember old u
        scalar_t u_old = u[i][y][x];

        // compute divergence of p
        scalar_t div_p;
        div_anisotropic(p, t, i, x, y, div_p);

        // gradient descent in the primal variable
        scalar_t u_new = u_old + tau * (div_p + lambda[i][y][x] * f[i][y][x]);

        // compute prox
        u_new /= (1 + tau * lambda[i][y][x]);

        // write back
        u[i][y][x] = u_new;

        // overrelaxation
        u__t_plus_1[i][y][x] = (1 + theta) * u_new - theta * u_old;
    }
}

template <typename scalar_t>
__global__ void TV_primal_u_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> t,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> f,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> p,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> u,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> lambda,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_u,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_u_,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_t,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_f,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_p,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_u_in,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_lambda,
    scalar_t tau,
    scalar_t theta)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int bs = f.size(0);
    const int ht = f.size(1);
    const int wd = f.size(2);

    if ((x >= wd) || (y >= ht))
        return;

    for (int i = 0; i < bs; ++i) { // i is batch index
        scalar_t g_u_new = grad_u[i][y][x] + (1 + theta) * grad_u_[i][y][x];
        scalar_t lam = lambda[i][y][x];
        grad_u_in[i][y][x] = -theta * grad_u_[i][y][x] + g_u_new / (1 + tau * lam);
        grad_f[i][y][x] = tau * lam / (1 + tau * lam) * g_u_new;
        grad_lambda[i][y][x] = tau / (1 + tau * lam) * (f[i][y][x] - u[i][y][x]) * g_u_new;
        scalar_t tmp = tau / (1 + tau * lam) * g_u_new;
        if (y > 0) {
            // div_p += t[b][y-1][x][0][0] * p[b][y-1][x][0] + t[b][y-1][x][1][0] * p[b][y-1][x][1];
            atomicAdd(&grad_p[i][y-1][x][0], -t[i][y-1][x][0][0] * tmp);
            atomicAdd(&grad_p[i][y-1][x][1], -t[i][y-1][x][1][0] * tmp);
            atomicAdd(&grad_t[i][y-1][x][0][0], -p[i][y-1][x][0] * tmp);
            atomicAdd(&grad_t[i][y-1][x][1][0], -p[i][y-1][x][1] * tmp);
        }
        if (y < ht - 1) {
            // div_p -= t[b][y][x][0][0] * p[b][y][x][0] + t[b][y][x][1][0] * p[b][y][x][1];
            atomicAdd(&grad_p[i][y][x][0], t[i][y][x][0][0] * tmp);
            atomicAdd(&grad_p[i][y][x][1], t[i][y][x][1][0] * tmp);
            atomicAdd(&grad_t[i][y][x][0][0], p[i][y][x][0] * tmp);
            atomicAdd(&grad_t[i][y][x][1][0], p[i][y][x][1] * tmp);
        }
        if (x < wd - 1) {
            // div_p -= t[b][y][x][0][1] * p[b][y][x][0] + t[b][y][x][1][1] * p[b][y][x][1];
            atomicAdd(&grad_p[i][y][x][0], t[i][y][x][0][1] * tmp);
            atomicAdd(&grad_p[i][y][x][1], t[i][y][x][1][1] * tmp);
            atomicAdd(&grad_t[i][y][x][0][1], p[i][y][x][0] * tmp);
            atomicAdd(&grad_t[i][y][x][1][1], p[i][y][x][1] * tmp);
        }
        if (x > 0) {
            // div_p += t[b][y][x-1][0][1] * p[b][y][x-1][0] + t[b][y][x-1][1][1] * p[b][y][x-1][1];
            atomicAdd(&grad_p[i][y][x-1][0], -t[i][y][x-1][0][1] * tmp);
            atomicAdd(&grad_p[i][y][x-1][1], -t[i][y][x-1][1][1] * tmp);
            atomicAdd(&grad_t[i][y][x-1][0][1], -p[i][y][x-1][0] * tmp);
            atomicAdd(&grad_t[i][y][x-1][1][1], -p[i][y][x-1][1] * tmp);
        }
    }
}

template<typename scalar_t>
__global__ void TGV_primal_v_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> t,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> p,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> q,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> v,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> v__t_plus_1,
    scalar_t tau,
    scalar_t theta)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int bs = v.size(0);
    const int ht = v.size(1);
    const int wd = v.size(2);

    if ((x >= wd) || (y >= ht))
        return;

    for (int i = 0; i < bs; ++i) { // i is batch index
        // remember old v
        scalar_t v_x_old = v[i][y][x][1];
        scalar_t v_y_old = v[i][y][x][0];

        // compute symmetric divergence of q
        scalar_t div_q_x, div_q_y;
        div_sym(q, p, t, i, x, y, div_q_x, div_q_y);

        // gradient descent in the primal variable
        scalar_t v_x = v_x_old + tau * div_q_x;
        scalar_t v_y = v_y_old + tau * div_q_y;

        // write back
        v[i][y][x][0] = v_y;
        v[i][y][x][1] = v_x;

        // overrelaxation
        v__t_plus_1[i][y][x][0] = (1 + theta) * v_y - theta * v_y_old;
        v__t_plus_1[i][y][x][1] = (1 + theta) * v_x - theta * v_x_old;
    }
}

template<typename scalar_t>
__global__ void TGV_primal_v_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> t,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> p,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_v,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_v_,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_t,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_p,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_q,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_v_in,
    scalar_t tau,
    scalar_t theta)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int bs = grad_v.size(0);
    const int ht = grad_v.size(1);
    const int wd = grad_v.size(2);

    if ((x >= wd) || (y >= ht))
        return;

    for (int i = 0; i < bs; ++i) { // i is batch index
        scalar_t g_v_new_x = grad_v[i][y][x][1] + (1 + theta) * grad_v_[i][y][x][1];
        scalar_t g_v_new_y = grad_v[i][y][x][0] + (1 + theta) * grad_v_[i][y][x][0];
        grad_v_in[i][y][x][0] = -theta * grad_v_[i][y][x][0] + g_v_new_y;
        grad_v_in[i][y][x][1] = -theta * grad_v_[i][y][x][1] + g_v_new_x;
        scalar_t tmp_x = tau * g_v_new_x;
        scalar_t tmp_y = tau * g_v_new_y;

        if (y > 0) {
            // div_q_yy_y
            atomicAdd(&grad_q[i][y-1][x][0], -tmp_y);
            // div_q_xy_y
            atomicAdd(&grad_q[i][y-1][x][1], -tmp_x);
        }
        if (y < ht - 1) {
            // div_q_yy_y
            atomicAdd(&grad_q[i][y][x][0], tmp_y);
            // div_q_xy_y
            atomicAdd(&grad_q[i][y][x][1], tmp_x);
            // div_q_y += t[b][y][x][0][0] * p[b][y][x][0] + t[b][y][x][1][0] * p[b][y][x][1];
            atomicAdd(&grad_p[i][y][x][0], t[i][y][x][0][0] * tmp_y);
            atomicAdd(&grad_p[i][y][x][1], t[i][y][x][1][0] * tmp_y);
            atomicAdd(&grad_t[i][y][x][0][0], p[i][y][x][0] * tmp_y);
            atomicAdd(&grad_t[i][y][x][1][0], p[i][y][x][1] * tmp_y);
        }
        if (x < wd - 1) {
            // div_q_xx_x
            atomicAdd(&grad_q[i][y][x][2], tmp_x);
            // div_q_xy_x
            atomicAdd(&grad_q[i][y][x][1], tmp_y);
            // div_q_x += t[b][y][x][0][1] * p[b][y][x][0] + t[b][y][x][1][1] * p[b][y][x][1];
            atomicAdd(&grad_p[i][y][x][0], t[i][y][x][0][1] * tmp_x);
            atomicAdd(&grad_p[i][y][x][1], t[i][y][x][1][1] * tmp_x);
            atomicAdd(&grad_t[i][y][x][0][1], p[i][y][x][0] * tmp_x);
            atomicAdd(&grad_t[i][y][x][1][1], p[i][y][x][1] * tmp_x);
        }
        if (x > 0) {
            // div_q_xx_x
            atomicAdd(&grad_q[i][y][x-1][2], -tmp_x);
            // div_q_xy_x
            atomicAdd(&grad_q[i][y][x-1][1], -tmp_y);
        }
    }
}

template<typename scalar_t>
__global__ void TGV_dual_p_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> t,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> u,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> v,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> p_unscaled,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> p,
    scalar_t sigma,
    scalar_t huber_delta)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int bs = v.size(0);
    const int ht = v.size(1);
    const int wd = v.size(2);

    if ((x >= wd) || (y >= ht))
        return;

    for (int i = 0; i < bs; ++i) { // i is batch index
        // compute gradient of u
        scalar_t grad_u_x, grad_u_y;
        grad_anisotropic(u, v, t, i, x, y, grad_u_x, grad_u_y);

        // gradient ascent in the dual variable
        scalar_t p_x = p[i][y][x][1] + sigma * grad_u_x;
        scalar_t p_y = p[i][y][x][0] + sigma * grad_u_y;
        p_x /= (sigma * huber_delta + 1);
        p_y /= (sigma * huber_delta + 1);

        // projection of p
        scalar_t scale_x = 1.0 / max(1.0, abs(p_x));
        scalar_t scale_y = 1.0 / max(1.0, abs(p_y));

        // write back
        p[i][y][x][0] = p_y * scale_y;
        p[i][y][x][1] = p_x * scale_x;
        p_unscaled[i][y][x][0] = p_y;
        p_unscaled[i][y][x][1] = p_x;
    }
}

template <typename scalar_t>
__global__ void TGV_dual_p_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> t,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> u,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> v,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> p_unscaled,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_p,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_p_in,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_u,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_v,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_t,
    scalar_t sigma,
    scalar_t huber_delta)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int bs = v.size(0);
    const int ht = v.size(1);
    const int wd = v.size(2);

    if ((x >= wd) || (y >= ht))
        return;

    for (int i = 0; i < bs; ++i) { // i is batch index
        scalar_t g_x = 0, g_y = 0;
        scalar_t p_x = p_unscaled[i][y][x][1], p_y = p_unscaled[i][y][x][0];
        if (abs(p_x) < 1) {
            g_x = grad_p[i][y][x][1];
        }
        if (abs(p_y) < 1) {
            g_y = grad_p[i][y][x][0];
        }
        grad_p_in[i][y][x][0] = g_y / (sigma * huber_delta + 1);
        grad_p_in[i][y][x][1] = g_x / (sigma * huber_delta + 1);
        scalar_t tmp_x = grad_p_in[i][y][x][1] * sigma;
        scalar_t tmp_y = grad_p_in[i][y][x][0] * sigma;

        const unsigned int xp = x + (x < u.size(2) - 1);
        const unsigned int yp = y + (y < u.size(1) - 1);

        scalar_t grad_u_x = u[i][y][xp] - u[i][y][x];
        scalar_t grad_u_y = u[i][yp][x] - u[i][y][x];
        scalar_t vx = (x < u.size(2) - 1) ? v[i][y][x][1] : 0;
        scalar_t vy = (y < u.size(1) - 1) ? v[i][y][x][0] : 0;
        scalar_t grad_e_x = grad_u_x - vx;
        scalar_t grad_e_y = grad_u_y - vy;

        if (x < wd - 1) {
            atomicAdd(&grad_t[i][y][x][1][1], grad_e_x * tmp_x);
            atomicAdd(&grad_t[i][y][x][0][1], grad_e_x * tmp_y);
            scalar_t tmp_ex = t[i][y][x][1][1] * tmp_x + t[i][y][x][0][1] * tmp_y;
            atomicAdd(&grad_u[i][y][x+1], tmp_ex);
            atomicAdd(&grad_u[i][y][x], -tmp_ex);
            atomicAdd(&grad_v[i][y][x][1], -tmp_ex);
        }
        if (y < ht - 1) {
            atomicAdd(&grad_t[i][y][x][1][0], grad_e_y * tmp_x);
            atomicAdd(&grad_t[i][y][x][0][0], grad_e_y * tmp_y);
            scalar_t tmp_ey = t[i][y][x][1][0] * tmp_x + t[i][y][x][0][0] * tmp_y;
            atomicAdd(&grad_u[i][y+1][x], tmp_ey);
            atomicAdd(&grad_u[i][y][x], -tmp_ey);
            atomicAdd(&grad_v[i][y][x][0], -tmp_ey);
        }
    }
}

template<typename scalar_t>
__global__ void TGV_dual_q_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> v,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> q_unscaled,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> q,
    scalar_t sigma,
    scalar_t alpha2)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int bs = v.size(0);
    const int ht = v.size(1);
    const int wd = v.size(2);

    if ((x >= wd) || (y >= ht))
        return;

    for (int i = 0; i < bs; ++i) { // i is batch index
        // compute symmetric gradient of v
        scalar_t grad_v_xx, grad_v_yy, grad_v_xy;
        grad_sym(v, i, x, y, grad_v_xx, grad_v_yy, grad_v_xy);

        // gradient ascent in the dual variable
        scalar_t q_xx = q[i][y][x][2] + sigma * grad_v_xx;
        scalar_t q_yy = q[i][y][x][0] + sigma * grad_v_yy;
        scalar_t q_xy = q[i][y][x][1] + sigma * grad_v_xy;

        // projection of p
        scalar_t norm = sqrt(q_xx * q_xx + q_yy * q_yy + 2 * q_xy * q_xy);
        scalar_t scale = 1.0 / max(1.0, norm / alpha2);

        // write back
        q[i][y][x][0] = q_yy * scale;
        q[i][y][x][1] = q_xy * scale;
        q[i][y][x][2] = q_xx * scale;

        q_unscaled[i][y][x][0] = q_yy;
        q_unscaled[i][y][x][1] = q_xy;
        q_unscaled[i][y][x][2] = q_xx;
    }
}

template<typename scalar_t>
__global__ void TGV_dual_q_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_q,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> q_unscaled,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_q_in,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_v,
    scalar_t sigma,
    scalar_t alpha2)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int bs = grad_q.size(0);
    const int ht = grad_q.size(1);
    const int wd = grad_q.size(2);

    if ((x >= wd) || (y >= ht))
        return;

    for (int i = 0; i < bs; ++i) { // i is batch index
        scalar_t g_xx = grad_q[i][y][x][2], g_xy = grad_q[i][y][x][1], g_yy = grad_q[i][y][x][0];
        scalar_t q_xx = q_unscaled[i][y][x][2], q_xy = q_unscaled[i][y][x][1], q_yy = q_unscaled[i][y][x][0];
        scalar_t norm_square = q_xx * q_xx + q_yy * q_yy + 2 * q_xy * q_xy;
        scalar_t norm = sqrt(norm_square);
        scalar_t g_xx_unscaled = 0, g_xy_unscaled = 0, g_yy_unscaled = 0;
        if (norm > alpha2) {
            g_xx_unscaled += (2 * q_xy * q_xy + q_yy * q_yy) / norm_square * g_xx;
            g_xy_unscaled -= 2 * q_xx * q_xy / norm_square * g_xx;
            g_yy_unscaled -= q_xx * q_yy / norm_square * g_xx;
            g_xx_unscaled -= q_xx * q_xy / norm_square * g_xy;
            g_xy_unscaled += (q_xx * q_xx + q_yy * q_yy) / norm_square * g_xy;
            g_yy_unscaled -= q_xy * q_yy / norm_square * g_xy;
            g_xx_unscaled -= q_xx * q_yy / norm_square * g_yy;
            g_xy_unscaled -= 2 * q_xy * q_yy / norm_square * g_yy;
            g_yy_unscaled += (q_xx * q_xx + 2 * q_xy * q_xy) / norm_square * g_yy;

            scalar_t beta_norm = alpha2 / norm;
            g_xx_unscaled *= beta_norm;
            g_xy_unscaled *= beta_norm;
            g_yy_unscaled *= beta_norm;
        } else {
            g_xx_unscaled = g_xx;
            g_xy_unscaled = g_xy;
            g_yy_unscaled = g_yy;
        }
        grad_q_in[i][y][x][0] = g_yy_unscaled;
        grad_q_in[i][y][x][1] = g_xy_unscaled;
        grad_q_in[i][y][x][2] = g_xx_unscaled;

        scalar_t tmp_xx = sigma * g_xx_unscaled;
        scalar_t tmp_xy = sigma * g_xy_unscaled * 0.5;
        scalar_t tmp_yy = sigma * g_yy_unscaled;

        if (x <  wd - 1) {
            atomicAdd(&grad_v[i][y][x+1][1], tmp_xx);
            atomicAdd(&grad_v[i][y][x][1], -tmp_xx);
            atomicAdd(&grad_v[i][y][x+1][0], tmp_xy);
            atomicAdd(&grad_v[i][y][x][0], -tmp_xy);
        }
        if (y < ht - 1) {
            atomicAdd(&grad_v[i][y+1][x][1], tmp_xy);
            atomicAdd(&grad_v[i][y][x][1], -tmp_xy);
            atomicAdd(&grad_v[i][y+1][x][0], tmp_yy);
            atomicAdd(&grad_v[i][y][x][0], -tmp_yy);
        }
    }
}

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
    double huber_delta)
{
    const auto ht = u.size(1);
    const auto wd = u.size(2);

    const dim3 blocks((wd + BLOCK - 1) / BLOCK,
                      (ht + BLOCK - 1) / BLOCK);

    const dim3 threads(BLOCK, BLOCK);

    auto u_t_plus_1 = u.clone();
    auto u__t_plus_1 = torch::empty_like(u_);
    auto v_t_plus_1 = v.clone();
    auto v__t_plus_1 = torch::empty_like(v_);
    auto p_t_plus_1 = p.clone();
    auto p_unscaled = torch::empty_like(p);
    auto q_t_plus_1 = q.clone();
    auto q_unscaled = torch::empty_like(q);

    //! dual step and projection on p
    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "TGV_dual_p_forward_kernel", ([&] {
        TGV_dual_p_forward_kernel<scalar_t><<<blocks, threads>>>(
            t.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            u_.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            v_.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            p_unscaled.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            p_t_plus_1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            (scalar_t)sigma,
            (scalar_t)huber_delta);
    }));

    //! dual step and prox on q
    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "TGV_dual_q_forward_kernel", ([&] {
        TGV_dual_q_forward_kernel<scalar_t><<<blocks, threads>>>(
            v_.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            q_unscaled.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            q_t_plus_1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            (scalar_t)sigma,
            (scalar_t)alpha_2);
    }));

    //! primal step on u
    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "TV_primal_u_forward_kernel", ([&] {
        TV_primal_u_forward_kernel<scalar_t><<<blocks, threads>>>(
            t.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            f.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            p_t_plus_1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            lambda.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            u_t_plus_1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            u__t_plus_1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            (scalar_t)tau,
            (scalar_t)theta);
    }));

    //! primal step on v
    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "TGV_primal_v_forward_kernel", ([&] {
        TGV_primal_v_forward_kernel<scalar_t><<<blocks, threads>>>(
            t.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            p_t_plus_1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            q_t_plus_1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            v_t_plus_1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            v__t_plus_1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            (scalar_t)tau,
            (scalar_t)theta);
    }));

    return {u_t_plus_1, u__t_plus_1, v_t_plus_1, v__t_plus_1, p_t_plus_1, p_unscaled, q_t_plus_1, q_unscaled};
}

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
    double huber_delta)
{
    const auto ht = f.size(1);
    const auto wd = f.size(2);

    const dim3 blocks((wd + BLOCK - 1) / BLOCK,
                      (ht + BLOCK - 1) / BLOCK);

    const dim3 threads(BLOCK, BLOCK);

    auto grad_u_in = torch::zeros_like(grad_u);
    auto grad_u__in = torch::zeros_like(grad_u_);
    auto grad_v_in = torch::zeros_like(grad_v);
    auto grad_v__in = torch::zeros_like(grad_v_);
    auto grad_p_in = torch::zeros_like(grad_p);
    auto grad_p_branch = torch::zeros_like(grad_p);
    auto grad_q_in = torch::zeros_like(grad_q);
    auto grad_q_branch = torch::zeros_like(grad_q);
    auto grad_t = torch::zeros_like(t);
    auto grad_f = torch::zeros_like(f);
    auto grad_lambda = torch::zeros_like(lambda);

    AT_DISPATCH_FLOATING_TYPES(grad_v.scalar_type(), "TGV_primal_v_backward_kernel", ([&] {
        TGV_primal_v_backward_kernel<scalar_t><<<blocks, threads>>>(
            t.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            p.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_v.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_v_.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_t.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            grad_p_branch.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_q_branch.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_v_in.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            (scalar_t)tau,
            (scalar_t)theta);
    }));

  AT_DISPATCH_FLOATING_TYPES(grad_v.scalar_type(), "TV_primal_u_backward_kernel", ([&] {
    TV_primal_u_backward_kernel<scalar_t><<<blocks, threads>>>(
        t.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        f.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        p.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        lambda.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_u.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_u_.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_t.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        grad_f.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_p_branch.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_u_in.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_lambda.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        (scalar_t)tau,
        (scalar_t)theta);
  }));

  grad_q += grad_q_branch;
  AT_DISPATCH_FLOATING_TYPES(grad_v.scalar_type(), "TGV_dual_q_backward_kernel", ([&] {
    TGV_dual_q_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_q.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        q_unscaled.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_q_in.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_v__in.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        (scalar_t)sigma,
        (scalar_t)alpha_2);
  }));

  grad_p += grad_p_branch;
  AT_DISPATCH_FLOATING_TYPES(grad_v.scalar_type(), "TGV_dual_p_backward_kernel", ([&] {
    TGV_dual_p_backward_kernel<scalar_t><<<blocks, threads>>>(
        t.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        u__in.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        v__in.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        p_unscaled.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_p.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_p_in.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_u__in.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_v__in.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_t.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        (scalar_t)sigma,
        (scalar_t)huber_delta);
  }));

  return {grad_u_in, grad_u__in, grad_v_in, grad_v__in, grad_p_in, grad_q_in, grad_t, grad_f, grad_lambda};
}