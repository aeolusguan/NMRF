import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from timm.models.layers import Mlp, DropPath, to_2tuple


def fourier_grid_embed(data, embed_dim):
    """data format: B[spatial dims]C

    Returns pos_embedding: same format with data
    """
    b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
    assert embed_dim % (2 * len(axis)) == 0

    # calculate fourier encoded positions in the range of [-1, 1], for all axis
    axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
    pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)

    num_bands = embed_dim // (2 * len(axis))
    pos = pos.unsqueeze(-1)
    freq_bands = torch.linspace(1., num_bands, num_bands, device=device, dtype=dtype)
    freq_bands = freq_bands[(*((None,) * (len(pos.shape) - 1)), Ellipsis)]
    pos_embedding = pos * freq_bands * math.pi
    pos_embedding = torch.cat([pos_embedding.sin(), pos_embedding.cos()], dim=-1)
    pos_embedding = rearrange(pos_embedding, '... n d -> ... (n d)')
    pos_embedding = repeat(pos_embedding, '... -> b ...', b=b)

    return pos_embedding


def fourier_coord_embed(coord, N_freqs, normalizer=3.14/512, logscale=True):
    """
    coord: [...]D
    returns:
        [...]dim, where dim=(2*N_freqs+1)*D
    """
    if logscale:
        freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs, device=coord.device)
    else:
        freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)
    coord = coord.unsqueeze(-1) * normalizer
    freq_bands = freq_bands[(*((None,) * (len(coord.shape) - 1)), Ellipsis)]
    f_coord = coord * freq_bands
    embed = torch.cat([f_coord.sin(), f_coord.cos(), coord], dim=-1)
    embed = rearrange(embed, '... d n -> ... (d n)')

    return embed


class MLP(nn.Module):
    """ Very simple multi-layer perception (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# Neural Message Passing along self edges of the NMRF graph.
class BasicAttention(nn.Module):
    """
    label representation:  [B, N, C]
    """
    def __init__(self, dim, qk_dim, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0., normalize_before=False):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be multiple times of heads {num_heads}'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.q, self.k, self.v = nn.Linear(qk_dim, dim, bias=True), nn.Linear(qk_dim, dim, bias=True), nn.Linear(dim, dim, bias=True)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.normalize_before = normalize_before

    def forward_pre(self, label_rep, abs_encoding):
        short_cut = label_rep
        label_rep = self.norm1(label_rep)
        q = k = torch.cat((label_rep, abs_encoding), dim=-1)

        # multi-head attention
        q, k, v = self.q(q), self.k(k), self.v(label_rep)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        attn = F.softmax(torch.einsum('bhid, bhjd -> bhij', q, k) * self.scale, dim=-1)
        attn = self.attn_drop(attn)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        out = torch.einsum('bhij, bhjd -> bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # residual connection
        label_rep = short_cut + self.proj_drop(self.proj(out))

        return label_rep

    def forward_post(self, label_rep, abs_encoding):
        short_cut = label_rep
        q = k = torch.cat((label_rep, abs_encoding), dim=-1)

        # multi-head attention
        q, k, v = self.q(q), self.k(k), self.v(label_rep)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        attn = F.softmax(torch.einsum('bhid, bhjd -> bhij', q, k) * self.scale, dim=-1)
        attn = self.attn_drop(attn)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        out = torch.einsum('bhij, bhjd -> bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # residual connection
        label_rep = short_cut + self.proj_drop(self.proj(out))
        label_rep = self.norm1(label_rep)

        return label_rep

    def forward(self, label_rep, abs_encoding):
        """
        label_rep: [B, N, C], embedding of the candidate label
        abs_encoding: [B, N, C'], encoding of the underlying absolute disparity of the candidate label
        Returns:
            [B, N, C], aggregated message from self edges
        """
        if self.normalize_before:
            return self.forward_pre(label_rep, abs_encoding)
        return self.forward_post(label_rep, abs_encoding)


class WindowAttention(nn.Module):
    """ Window based multi-head positional sensitive self attention (W-MSA).
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        shift_size (int): Shift size for SW-MSA.
        num_heads (int): Number of attention heads.
        qk_scale (float | None, optional): Override a default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
    """

    def __init__(self, dim, window_size, shift_size, num_heads, qk_scale=None, attn_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.shift_size = shift_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # define a parameter table of relative position bias
        self.relative_position_enc_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), dim*3))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def window_partition(self, x):
        """
        x: [B, H, W, N, C]
        Returns:
            (B*num_windows, num_heads, window_size*window_size*N, head_dim]
        """
        x = rearrange(x, 'b (i hs) (j ws) n (h d) -> (b i j) h (hs ws n) d',
                      hs=self.window_size[0], ws=self.window_size[1], h=self.num_heads)
        return x

    @staticmethod
    def gen_window_attn_mask(window_size, device=torch.device('cuda')):
        """
        Generating attention mask to prevent message passing along self edges.

        Args:
            window_size (tuple[int]): The height, width, and depth (number of candidates) of attention window
        """
        idx = torch.arange(0, window_size[0] * window_size[1], dtype=torch.float32, device=device).view(-1, 1)
        idx = idx.expand(window_size[0] * window_size[1], window_size[2]).flatten()
        attn_mask = idx.unsqueeze(-1) - idx.unsqueeze(0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask != 0, 0.0)
        attn_mask.fill_diagonal_(0.0)
        return attn_mask

    @staticmethod
    def gen_shift_window_attn_mask(input_resolution, window_size, shift_size, device=torch.device('cuda')):
        """
        Generating attention mask for shifted window attention, modified from SWin Transformer.

        Args:
            input_resolution (tuple[int]): The height and width of input
            window_size (tuple[int]): The height, width and depth (number of candidates) of window
            shift_size (int): shift size for SW-MSA.
        """
        H, W = input_resolution
        img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
        h_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = rearrange(img_mask, 'b (h hs) (w ws) c -> (b h w) (hs ws) c', hs=window_size[0], ws=window_size[1])
        mask_windows = mask_windows.squeeze(-1)  # [num_windows, window_size*window_size]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, float('0.0'))
        attn_mask = repeat(attn_mask, 'b h w -> b (h h2) (w w2)', h2=window_size[2], w2=window_size[2])
        attn_mask = attn_mask + WindowAttention.gen_window_attn_mask(window_size, device).unsqueeze(0)
        return attn_mask

    def forward(self, qkv, attn_mask):
        """
        qkv:   [B,H,W,N,3*C]
        mask:  [num_windows, window_size*window_size, window_size*window_size]
        Returns:
            BHWNC
        """
        bs, ht, wd, n, _ = qkv.shape
        if self.shift_size > 0:
            qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        query, key, value = qkv.chunk(3, dim=-1)
        q = self.window_partition(query)
        k = self.window_partition(key)
        v = self.window_partition(value)

        # positional embedding
        rpe = self.relative_position_enc_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], self.num_heads, -1)
        rpe = repeat(rpe, 'i j h c -> (i hs) (j ws) h c', hs=n, ws=n)
        q_embed, k_embed, v_embed = rpe.chunk(3, dim=-1)

        # window attention
        q = q * self.scale
        q_embed = q_embed * self.scale
        qk = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        qr = torch.einsum('bhic,ijhc->bhij', q, k_embed)
        kr = torch.einsum('bhjc,ijhc->bhij', k, q_embed)
        attn = qk + qr + kr

        attn = rearrange(attn, '(b j) h m n -> b j h m n', b=bs)

        window_size = self.window_size

        if attn_mask is not None:
            attn = attn + attn_mask[None, :, None, :, :]
        attn = rearrange(attn, 'b j h m n -> (b j) h m n')

        attn = F.softmax(attn, dim=-1, dtype=attn.dtype)

        attn = self.attn_drop(attn)

        x = attn @ v + torch.einsum('bhij,ijhc->bhic', attn, v_embed)
        x = rearrange(x, '(b i j) h (hs ws n) d -> b (i hs) (j ws) n (h d)', i=ht//window_size[0],
                      j=wd//window_size[1], hs=window_size[0], ws=window_size[1])

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, shift_size={self.shift_size}, num_heads={self.num_heads}'


class SwinNMP(nn.Module):
    r"""Swin Message Passing Block.

    Args:
        dim (int): Number of input channels.
        qkv_dim (int): Number of input token channels
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, qkv_dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, normalize_before=False):
        super().__init__()
        self.dim = dim
        self.qkv_dim = qkv_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.qkv = nn.Linear(qkv_dim, 3*dim, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), shift_size=shift_size, num_heads=num_heads,
            qk_scale=qk_scale, attn_drop=attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
        self.normalize_before = normalize_before

    def get_qkv_input(self, label_rep, abs_encoding):
        label_rep = self.norm1(label_rep) if self.normalize_before else label_rep
        # concat label embedding with absolute disparity embedding
        x = torch.cat((label_rep, abs_encoding), dim=-1)
        x = rearrange(x, '(b h w) n c -> b h w n c', h=self.H, w=self.W)
        return x

    def forward_pre(self, label_rep, abs_encoding, attn_mask):
        shortcut = label_rep

        qkv_input = self.get_qkv_input(label_rep, abs_encoding)

        qkv = self.qkv(qkv_input)
        # window attention
        msg = self.attn(qkv, attn_mask)
        msg = self.proj_drop(self.proj(msg))

        msg = rearrange(msg, 'b h w n c -> (b h w) n c')
        x = shortcut + self.drop_path(msg)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def forward_post(self, label_rep, abs_encoding, attn_mask):
        shortcut = label_rep

        qkv_input = self.get_qkv_input(label_rep, abs_encoding)

        qkv = self.qkv(qkv_input)
        # window attention
        msg = self.attn(qkv, attn_mask)
        msg = self.proj_drop(self.proj(msg))

        msg = rearrange(msg, 'b h w n c -> (b h w) n c')
        x = shortcut + self.drop_path(msg)
        x = self.norm1(x)
        x = x + self.drop_path(self.mlp(x))
        x = self.norm2(x)

        return x

    def forward(self, label_rep, abs_encoding, attn_mask):
        """
        label_rep: [B*H*W, N, C], embedding of the candidate label
        abs_encoding: [B*H*W, N, C'], encoding of the underlying
            absolute disparity of the candidate label
        attn_mask: [num_windows, window_size_h*window_size_w, window_size_h*window_size_w],
            attention mask to prevent message passing along self edges.
        """
        if self.normalize_before:
            return self.forward_pre(label_rep, abs_encoding, attn_mask)
        return self.forward_post(label_rep, abs_encoding, attn_mask)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, qkv_dim={self.qkv_dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, normalize_before={self.normalize_before}"


class CSWinAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, num_heads=8, qk_scale=None, attn_drop=0.):
        """Attention within cross-shaped windows.
        """
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.idx = idx
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            raise ValueError(f"ERROR MODE {idx}")
        self.H_sp = H_sp
        self.W_sp = W_sp

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        x = rearrange(x, 'b (i hs) (j ws) n (h d) -> (b i j) h (hs ws n) d', hs=self.H_sp, ws=self.W_sp, h=self.num_heads)
        return x

    def get_rpe(self, x, func):
        n = x.shape[3]
        x = rearrange(x, 'b (i hs) (j ws) n d -> (b i j n) d hs ws', hs=self.H_sp, ws=self.W_sp)
        rpe = func(x.contiguous())  # B', C, H', W'
        rpe = torch.sum(rearrange(rpe, '(b n) d hs ws -> b n d hs ws', n=n), dim=1, keepdim=True)

        # remove the positional embedding of self edges
        mask = torch.eye(n, device=x.device, dtype=torch.bool)
        mask = (~mask).float() * (-1)
        tmp = rearrange(x, '(b n) d hs ws -> b n d hs ws', n=n) * ((self.get_v.weight[:, 0, 1, 1])[None, None, :, None, None])
        tmp = torch.einsum('jk,bkdhw->bjdhw', mask, tmp)
        rpe = rpe + tmp

        rpe = rearrange(rpe, 'b n (h d) hs ws -> b h (hs ws n) d', h=self.num_heads)

        x = rearrange(x, '(b n) (h d) hs ws -> b h (hs ws n) d', n=n, h=self.num_heads)
        return x, rpe

    def forward(self, query, key, value):
        """
        query: BHWNC
        key:   BHWNC
        value: BHWNC'
        Returns:
            BHWNC
        """
        _, ht, wd, dd, _ = query.shape

        idx = self.idx
        if idx == -1:
            H_sp, W_sp = ht, wd
        elif idx == 0:
            H_sp, W_sp = ht, self.split_size
        elif idx == 1:
            H_sp, W_sp = self.split_size, wd
        else:
            raise RuntimeError(f"ERROR MODE `{idx}` in forward")
        self.H_sp = H_sp
        self.W_sp = W_sp

        ### padding for split window
        H_pad = (self.H_sp - ht % self.H_sp) % self.H_sp
        W_pad = (self.W_sp - wd % self.W_sp) % self.W_sp
        top_pad = H_pad // 2
        down_pad = H_pad - top_pad
        left_pad = W_pad // 2
        right_pad = W_pad - left_pad
        ht_ = ht + H_pad
        wd_ = wd + W_pad

        q = F.pad(query, (0, 0, 0, 0, left_pad, right_pad, top_pad, down_pad))
        k = F.pad(key, (0, 0, 0, 0, left_pad, right_pad, top_pad, down_pad))
        v = F.pad(value, (0, 0, 0, 0, left_pad, right_pad, top_pad, down_pad))

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, rpe = self.get_rpe(v, self.get_v)

        ### Local attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        window_size = (self.H_sp, self.W_sp, dd)
        attn = attn + WindowAttention.gen_window_attn_mask(window_size, attn.device)[None, None, ...]

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)

        attn = self.attn_drop(attn)

        x = (attn @ v) + rpe
        x = rearrange(x, '(b i j) h (hs ws n) d -> b (i hs) (j ws) n (h d)', i=ht_//self.H_sp, j=wd_//self.W_sp, hs=self.H_sp, ws=self.W_sp)
        x = x[:, top_pad:ht+top_pad, left_pad:wd+left_pad, :, :]

        return x


class CSWinNMP(nn.Module):

    def __init__(self, dim, qk_dim, v_dim, patches_resolution, num_heads,
                 split_size=7, mlp_ratio=4., qk_scale=None,
                 attn_drop=0., proj_drop=0., drop_path=0., dropout=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, normalize_before=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = patches_resolution
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.v_dim = v_dim

        self.q, self.k, self.v = nn.Linear(qk_dim, dim, bias=True), nn.Linear(qk_dim, dim, bias=True), nn.Linear(v_dim, dim, bias=True)
        self.norm1 = norm_layer(dim)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attns = nn.ModuleList([
            CSWinAttention(
                dim//2, resolution=self.patches_resolution, idx=i,
                split_size=split_size, num_heads=num_heads//2, qk_scale=qk_scale,
                attn_drop=attn_drop)
            for i in range(2)])
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=dropout)
        self.norm2 = norm_layer(dim)

        self.H = None
        self.W = None
        self.normalize_before = normalize_before

    def get_qkv(self, tgt, context):
        tgt2 = self.norm1(tgt) if self.normalize_before else tgt
        tgt2 = rearrange(tgt2, '(b h w) n c -> b h w n c', h=self.H, w=self.W)
        if context is not None:
            q = k = torch.cat((tgt2, context), dim=-1)
        else:
            q = k = tgt2

        if self.v_dim > self.dim:
            pe = fourier_grid_embed(tgt2[:, :, :, 0, :], embed_dim=(self.v_dim - self.dim))
            pe = pe.unsqueeze(-2).repeat(1, 1, 1, tgt2.shape[-2], 1)
            v = torch.cat((tgt2, pe), dim=-1)
        else:
            v = tgt2

        return q, k, v

    def forward_pre(self, tgt, context):
        shortcut = tgt

        q, k, v = self.get_qkv(tgt, context)
        query, key, value = self.q(q), self.k(k), self.v(v)

        # cross shaped window attention
        x1 = self.attns[0](query[..., :self.dim // 2], key[..., :self.dim // 2], value[..., :self.dim // 2])
        x2 = self.attns[1](query[..., self.dim // 2:], key[..., self.dim // 2:], value[..., self.dim // 2:])
        msg = torch.cat([x1, x2], dim=-1).reshape(tgt.shape)
        msg = self.proj_drop(self.proj(msg))
        x = shortcut + self.drop_path(msg)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward_post(self, tgt, context):
        shortcut = tgt

        q, k, v = self.get_qkv(tgt, context)
        query, key, value = self.q(q), self.k(k), self.v(v)

        # cross shaped window attention
        x1 = self.attns[0](query[..., :self.dim // 2], key[..., :self.dim // 2], value[..., :self.dim // 2])
        x2 = self.attns[1](query[..., self.dim // 2:], key[..., self.dim // 2:], value[..., self.dim // 2:])
        msg = torch.cat([x1, x2], dim=-1).reshape(tgt.shape)
        msg = self.proj_drop(self.proj(msg))
        x = shortcut + self.drop_path(msg)
        x = self.norm1(x)
        x = x + self.drop_path(self.mlp(x))
        x = self.norm2(x)
        return x

    def forward(self, seed_rep, context):
        """
        seed_rep: [BHW,N,C], embedding of the label seed
        context: BHWNC', visual context of the label seed
        """
        if self.normalize_before:
            return self.forward_pre(seed_rep, context)
        return self.forward_post(seed_rep, context)


class Propagation(nn.Module):
    """Label seed propagation"""
    def __init__(self, embed_dim, cost_group, layers, norm=None, return_intermediate=False):
        super().__init__()
        self.cost_encoder = nn.Sequential(
            nn.Linear(cost_group*9, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.proj = nn.Linear(embed_dim+31, embed_dim, bias=False)
        self.embed_dim = embed_dim
        self.layers = layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    @staticmethod
    def sample_cost(cost_volume, label_seed):
        """
        cost_volume: [B*H*W,G,D], groupwise cost volume
        label_seed: [B*H*W,num_seed], integer disparity modals
        return:
            [B*H*W,num_seed,G*9], sampled cost values of the label seed
        """
        G, nd = cost_volume.shape[-2:]
        num_seed = label_seed.shape[1]
        offset = torch.arange(-4, 5, device=cost_volume.device, dtype=label_seed.dtype)
        idx = label_seed[..., None] + offset.view(1, 1, -1)  # [B*H*W,num_seed,9]
        idx = torch.clamp(idx, min=0, max=nd-1)
        idx = idx.reshape(-1, 1, 9*num_seed).repeat(1, G, 1)
        cost = torch.gather(cost_volume, dim=-1, index=idx)
        cost = rearrange(cost, 'b h (n c) -> b n (h c)', n=num_seed)
        return cost

    def forward(self, cost_volume, label_seed, context):
        """
        cost_volume: B*H*W,G,D, groupwise cost volume
        coord_seed: [B*H*W, num_seed], integer disparity modals
        context:    [ B,H,W,C], visual context
        """
        # extract cost of label seeds
        N = label_seed.shape[-1]
        cost = Propagation.sample_cost(cost_volume, label_seed)
        cost_feat = self.cost_encoder(cost)
        label_seed = label_seed.float()
        disp_encoding = fourier_coord_embed(label_seed.unsqueeze(-1), N_freqs=15, normalizer=3.14/64)
        # concat & proj
        sample_embed = self.proj(torch.cat((cost_feat, disp_encoding), dim=-1))

        context = repeat(context, 'b h w c -> b h w n c', n=N)
        intermediate = []
        for layer in self.layers:
            sample_embed = layer(sample_embed, context)
            if self.return_intermediate:
                intermediate.append(self.norm(sample_embed))

        if self.norm is not None:
            sample_embed = self.norm(sample_embed)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(sample_embed)

        if self.return_intermediate:
            return torch.stack(intermediate), label_seed

        return sample_embed.unsqueeze(0), label_seed


class Inference(nn.Module):
    """Neural MRF Inference"""
    def __init__(self, cost_group, dim, layers, norm, return_intermediate=False):
        super().__init__()

        self.ffn = Mlp(dim+cost_group, dim, dim)
        self.dim = dim
        self.layers = layers
        self.norm = norm
        self.cost_group = cost_group
        self.return_intermediate = return_intermediate

    @staticmethod
    def sample_fmap(fmap, disp_sample, radius=4):
        """
        fmap: [B, C, H, W]
        disp_sample: tensor of dim [B*H*W, num_disp], disparity samples
        radius(int): 2*radius+1 samples will be sampled for each sample
        return:
            sampled fmap feature of dim [B, C, H, W, num_disp*(2*radius+1)]
        """
        bs, _, ht, wd = fmap.shape
        num_disp = disp_sample.shape[1]
        device = fmap.device
        with torch.no_grad():
            offset = torch.arange(-radius, radius+1, dtype=disp_sample.dtype, device=disp_sample.device).view(1, 1, -1)
            grid_x = disp_sample[..., None] + offset  # [B*H*W, num_disp, 2*r+1]
            grid_x = grid_x.reshape(bs, ht, wd, -1)  # [B, H, W, num_disp*(2*r+1)]
            grid_y = torch.zeros_like(grid_x)
            xs = torch.arange(0, wd, device=device, dtype=torch.float32).view(1, wd).expand(ht, wd)
            ys = torch.arange(0, ht, device=device, dtype=torch.float32).view(ht, 1).expand(ht, wd)
            grid = torch.stack((xs, ys), dim=-1).reshape(1, ht, wd, 1, 2)
            grid = grid + torch.stack((-grid_x, grid_y), dim=-1)  # [B, H, W, num_disp*(2*r+1), 2]
            grid[..., 0] = 2 * grid[..., 0].clone() / (wd - 1) - 1
            grid[..., 1] = 2 * grid[..., 1].clone() / (ht - 1) - 1
            grid = grid.reshape(bs, ht, -1, 2)  # [B, H, W*num_disp*(2*r+1), 2]
        samples = F.grid_sample(fmap, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return samples.reshape(bs, -1, ht, wd, num_disp*(2*radius+1))

    def corr(self, fmap1, warped_fmap2, num_disp):
        """
        fmap1: [B, C, H, W]
        warped_fmap2: [B, C, H, W, num_disp]
        Returns:
            local cost: [B*H*W, num_disp, G]
        """
        fmap1 = rearrange(fmap1, 'b (g d) h w -> b g d h w', g=self.cost_group)
        warped_fmap2 = rearrange(warped_fmap2, 'b (g d) h w n -> b g d h w n', g=self.cost_group)
        corr = (fmap1.unsqueeze(-1) * warped_fmap2).mean(dim=2)  # [B, G, H, W, num_disp]
        corr = rearrange(corr, 'b g h w n -> (b h w) n g', n=num_disp)
        return corr

    def forward(self, labels, fmap1, fmap2, fmap1_gw, fmap2_gw):
        """
        labels: [B*H*W,num_disp], candidate labels (disparity)
        fmap1: [B,C,H,W]
        fmap2: [B,C,H,W]
        fmap1_gw: [B,C,H,W]
        fmap2_gw: [B,C,H,W]
        Returns:
            [B, H, W]
        """
        bs, _, ht, wd = fmap1.shape
        N = labels.shape[-1]
        device = labels.device
        warped_fmap2_gw = self.sample_fmap(fmap2_gw, labels, radius=0)  # [B,C,H,W,N]
        corr = self.corr(fmap1_gw, warped_fmap2_gw, N)  # [B*H*W,N,G]
        warped_fmap2 = self.sample_fmap(fmap2, labels, radius=0)  # [B,C,H,W,N]
        fmap1 = repeat(fmap1, 'b c h w -> b c h w n', n=N)
        feat_concat = torch.cat((fmap1, warped_fmap2), dim=1)
        feat_concat = rearrange(feat_concat, 'b c h w n -> (b h w) n c')
        label_rep = self.ffn(torch.cat((feat_concat, corr), dim=-1))

        abs_encoding = fourier_coord_embed(labels.unsqueeze(-1), N_freqs=15, normalizer=3.14/64)

        # pad input to multiple times of window_size (assume all swin blocks have the same window size)
        window_size = self.layers[0].window_size
        H_pad = (window_size - ht % window_size) % window_size
        W_pad = (window_size - wd % window_size) % window_size
        top_pad = H_pad // 2
        down_pad = H_pad - top_pad
        left_pad = W_pad // 2
        right_pad = W_pad - left_pad
        ht_ = ht + H_pad
        wd_ = wd + W_pad

        if H_pad > 0 or W_pad > 0:
            label_rep = rearrange(label_rep, '(b h w) n c -> b h w n c', h=ht, w=wd)
            label_rep = F.pad(label_rep, (0, 0, 0, 0, left_pad, right_pad, top_pad, down_pad))
            label_rep = rearrange(label_rep, 'b h w n c -> (b h w) n c')
            abs_encoding = rearrange(abs_encoding, '(b h w) n c -> b h w n c', h=ht, w=wd)
            abs_encoding = F.pad(abs_encoding, (0, 0, 0, 0, left_pad, right_pad, top_pad, down_pad))
            abs_encoding = rearrange(abs_encoding, 'b h w n c -> (b h w) n c')

        # hack implementation to cache attention mask
        window_size = (window_size, window_size, N)
        window_attn_mask = WindowAttention.gen_window_attn_mask(window_size, device=device)[None]
        attn_mask = [window_attn_mask]
        if len(self.layers) >= 2:
            shift_size = self.layers[1].shift_size
            input_resolution = (ht_, wd_)
            shifted_window_attn_mask = WindowAttention.gen_shift_window_attn_mask(
                input_resolution, window_size, shift_size, device
            )
            attn_mask.append(shifted_window_attn_mask)

        intermediate = []
        return_intermediate = self.return_intermediate and self.training
        for idx, layer in enumerate(self.layers):
            label_rep = layer(label_rep, abs_encoding, attn_mask[idx%2], ht_, wd_)
            if return_intermediate:
                label_rep_ = rearrange(label_rep, '(b h w) n c -> b h w n c', h=ht_, w=wd_)
                label_rep_ = label_rep_[:, top_pad:top_pad+ht, left_pad:left_pad+wd, :, :]
                label_rep_ = rearrange(label_rep_, 'b h w n c -> (b h w) n c')
                intermediate.append(self.norm(label_rep_))

        label_rep = rearrange(label_rep, '(b h w) n c -> b h w n c', h=ht_, w=wd_)
        label_rep = label_rep[:, top_pad:top_pad+ht, left_pad:left_pad+wd, :, :]
        label_rep = rearrange(label_rep, 'b h w n c -> (b h w) n c')
        if self.norm is not None:
            label_rep = self.norm(label_rep)
            if return_intermediate:
                intermediate.pop()
                intermediate.append(label_rep)

        if return_intermediate:
            return torch.stack(intermediate)

        return label_rep.unsqueeze(0)


class Refinement(Inference):
    @staticmethod
    def gen_shift_window_attn_mask(input_resolution, window_size, shift_size, device=torch.device('cuda')):
        """
        input_resolution (tuple[int]): The height and width of input
        window_size (tuple[int]): The height, width and depth of window
        shift_size (int): Shift size for SW-MSA.
        """
        H, W = input_resolution
        img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
        h_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = rearrange(img_mask, 'b (h hs) (w ws) c -> (b h w) (hs ws) c', hs=window_size[0], ws=window_size[1])
        mask_windows = mask_windows.squeeze(-1)  # [num_windows, window_size*window_size]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, float('0.0'))
        return attn_mask

    def forward(self, labels, fmap1, fmap2, fmap1_gw, fmap2_gw):
        """
        labels: [B,H,W]
        fmap1: [B,C,H,W]
        fmap2: [B,C,H,W]
        fmap1_gw: [B,C,H,W]
        fmap2_gw: [B,C,H,W]
        """
        bs, _, ht, wd = fmap1.shape
        device = labels.device
        labels = labels.reshape(-1, 1)
        warped_fmap2_gw = self.sample_fmap(fmap2_gw, labels, radius=0)  # [B,C,H,W,1]
        corr = self.corr(fmap1_gw, warped_fmap2_gw, 1)  # [B*H*W,1,G]
        warped_fmap2 = self.sample_fmap(fmap2, labels, radius=0).squeeze(-1)  # [B,C,H,W]
        feat_concat = torch.cat((fmap1, warped_fmap2), dim=1)
        feat_concat = rearrange(feat_concat, 'b c h w -> (b h w) 1 c')
        label_rep = self.ffn(torch.cat((feat_concat, corr), dim=-1))

        abs_encoding = fourier_coord_embed(labels.unsqueeze(-1), N_freqs=15, normalizer=3.14 / 128)

        # pad input to multiple times of window_size (assume all swin blocks have the same window size)
        window_size = self.layers[0].window_size
        H_pad = (window_size - ht % window_size) % window_size
        W_pad = (window_size - wd % window_size) % window_size
        top_pad = H_pad // 2
        down_pad = H_pad - top_pad
        left_pad = W_pad // 2
        right_pad = W_pad - left_pad
        ht_ = ht + H_pad
        wd_ = wd + W_pad

        if H_pad > 0 or W_pad > 0:
            label_rep = rearrange(label_rep, '(b h w) n c -> b h w n c', h=ht, w=wd)
            label_rep = F.pad(label_rep, (0, 0, 0, 0, left_pad, right_pad, top_pad, down_pad))
            label_rep = rearrange(label_rep, 'b h w n c -> (b h w) n c')
            abs_encoding = rearrange(abs_encoding, '(b h w) n c -> b h w n c', h=ht, w=wd)
            abs_encoding = F.pad(abs_encoding, (0, 0, 0, 0, left_pad, right_pad, top_pad, down_pad))
            abs_encoding = rearrange(abs_encoding, 'b h w n c -> (b h w) n c')

        # hack implementation to cache attention mask
        window_size = (window_size, window_size)
        attn_mask = [None]
        if len(self.layers) >= 2:
            shift_size = self.layers[1].shift_size
            input_resolution = (ht_, wd_)
            shifted_window_attn_mask = self.gen_shift_window_attn_mask(
                input_resolution, window_size, shift_size, device
            )
            attn_mask.append(shifted_window_attn_mask)

        intermediate = []
        return_intermediate = self.return_intermediate and self.training
        for idx, layer in enumerate(self.layers):
            label_rep = layer(label_rep, abs_encoding, attn_mask[idx % 2], ht_, wd_)
            if return_intermediate:
                label_rep_ = rearrange(label_rep, '(b h w) n c -> b h w n c', h=ht_, w=wd_)
                label_rep_ = label_rep_[:, top_pad:top_pad + ht, left_pad:left_pad + wd, :, :]
                label_rep_ = rearrange(label_rep_, 'b h w n c -> (b h w) n c')
                intermediate.append(self.norm(label_rep_))

        label_rep = rearrange(label_rep, '(b h w) n c -> b h w n c', h=ht_, w=wd_)
        label_rep = label_rep[:, top_pad:top_pad + ht, left_pad:left_pad + wd, :, :]
        label_rep = rearrange(label_rep, 'b h w n c -> (b h w) n c')
        if self.norm is not None:
            label_rep = self.norm(label_rep)
            if return_intermediate:
                intermediate.pop()
                intermediate.append(label_rep)

        if return_intermediate:
            return torch.stack(intermediate).squeeze(-2)

        return label_rep.unsqueeze(0).squeeze(-2)


class PropagationLayer(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, context_dim, split_size, n_heads,
                 activation="gelu", attn_drop=0., proj_drop=0., drop_path=0., dropout=0., normalize_before=False):
        super().__init__()

        # self attention
        act_layer = _get_activation_fn(activation)
        # concat seed embedding with visual context when linearly projecting to
        # query and key since visually similar pixel tends to have coherent disparities
        qk_dim = embed_dim + context_dim
        v_dim = embed_dim
        self.nmp = CSWinNMP(embed_dim, qk_dim, v_dim, patches_resolution=224//8, num_heads=n_heads,
                            split_size=split_size, mlp_ratio=mlp_ratio, attn_drop=attn_drop,
                            proj_drop=proj_drop, drop_path=drop_path, dropout=dropout, act_layer=act_layer,
                            normalize_before=normalize_before)

    def forward(self, tgt, context):
        """
        tgt:        BHW,N,C
        context:    B,H,W,N,C
        Returns:
            BHW,N,C
        """
        # self attention
        self.nmp.H, self.nmp.W = context.shape[1:3]
        tgt = self.nmp(tgt, context=context)
        return tgt


class InferenceLayer(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, window_size, shift_size, n_heads,
                 activation="gelu", attn_drop=0., proj_drop=0., drop_path=0., dropout=0., normalize_before=False):
        super().__init__()

        # self attention
        act_layer = _get_activation_fn(activation)
        qk_dim = embed_dim + 31
        self.window_size = window_size
        self.shift_size = shift_size
        # attend to proposals of the same pixel to suppress non-accurate proposals
        self.self_nmp = BasicAttention(embed_dim, qk_dim, n_heads, attn_drop=attn_drop, proj_drop=proj_drop,
                                       drop_path=drop_path, dropout=dropout, normalize_before=normalize_before)
        # attend to neighbor pixels to extract feature
        self.nmp = SwinNMP(embed_dim, qk_dim, num_heads=n_heads, window_size=window_size,
                           shift_size=shift_size, mlp_ratio=mlp_ratio, attn_drop=attn_drop,
                           drop_path=drop_path, drop=dropout, act_layer=act_layer, normalize_before=normalize_before)

    def forward(self, tgt, abs_encoding, attn_mask, ht, wd):
        """
        tgt: B*H*W,N,C
        abs_encoding: B*H*W,N,C
        """
        tgt = self.self_nmp(tgt, abs_encoding=abs_encoding)
        self.nmp.H, self.nmp.W = ht, wd
        tgt = self.nmp(tgt, abs_encoding=abs_encoding, attn_mask=attn_mask)
        return tgt


class RefinementLayer(nn.Module):
    def __init__(self, dim, mlp_ratio, window_size, shift_size, n_heads,
                 activation="gelu", attn_drop=0., proj_drop=0., drop_path=0., dropout=0., normalize_before=False):
        super().__init__()

        act_layer = _get_activation_fn(activation)
        qk_dim = dim + 31
        self.window_size = window_size
        self.shift_size = shift_size
        self.nmp = SwinNMP(dim, qk_dim, num_heads=n_heads, window_size=window_size,
                           shift_size=shift_size, mlp_ratio=mlp_ratio, attn_drop=attn_drop,
                           drop_path=drop_path, drop=dropout, act_layer=act_layer, normalize_before=normalize_before)

    def forward(self, tgt, abs_encoding, attn_mask, ht, wd):
        """
        tgt:        [B*H*W,1,C]
        context:    [B*H*W,1,C]
        """
        self.nmp.H, self.nmp.W = ht, wd
        tgt = self.nmp(tgt, abs_encoding=abs_encoding, attn_mask=attn_mask)
        return tgt


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
