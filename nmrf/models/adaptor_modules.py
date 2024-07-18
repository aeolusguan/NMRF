from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from timm.models.layers import DropPath
from ops.modules import MSDeformAttn


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs_dn(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 4, w // 4),
                                      (h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = spatial_shapes.new_zeros((1,))
    reference_points = get_reference_points([(h // 4, w // 4)], x.device)
    return reference_points, spatial_shapes, level_start_index


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x).flatten(2).transpose(1, 2)
        return x


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=8, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class ConvStem(nn.Module):
    def __init__(self, inplanes=64, out_channels=256, norm_layer=nn.InstanceNorm2d, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.fc = nn.Conv2d(inplanes, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        def _inner_forward(x):
            c = self.stem(x)
            c = self.fc(c)

            bs, dim, _, _ = c.shape
            c = c.view(bs, dim, -1).transpose(1, 2)  # 4s

            return c

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs


class DeformNeck(nn.Module):
    def __init__(self, dim, in_channel_list, num_heads=8, n_points=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), drop=0., drop_path=0.,
                 with_cffn=True, cffn_ratio=0.25, deform_ratio=1.0, with_cp=False):
        super().__init__()
        self.stem = ConvStem(inplanes=64, out_channels=dim)
        self.dim = dim

        self.extractors = nn.ModuleList([
            Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                      norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                      cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
            for _ in range(4)
        ])

        assert len(in_channel_list) == 4
        self.fcs = nn.ModuleList([nn.Sequential(*[
            norm_layer(in_channel),
            nn.Linear(in_channel, dim)])
            for in_channel in in_channel_list
        ])

    def forward(self, image, features):
        deform_inputs_scales = deform_inputs_dn(image)

        c = self.stem(image)

        _H, _W = image.shape[-2:]
        H, W = _H // 4, _W // 4

        assert len(self.extractors) == len(features)

        for idx, feat in enumerate(features):
            bs, dim, _, _ = feat.shape
            feat = feat.view(bs, dim, -1).transpose(1, 2)
            feat = self.fcs[idx](feat)
            c = self.extractors[idx](query=c, reference_points=deform_inputs_scales[0],
                                     feat=feat, spatial_shapes=deform_inputs_scales[1][idx:idx+1],
                                     level_start_index=deform_inputs_scales[2], H=H, W=W)

        bs, _, dim = c.shape
        c = c.transpose(1, 2).reshape(bs, dim, H, W)

        return c