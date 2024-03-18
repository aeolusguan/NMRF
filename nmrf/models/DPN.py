import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange

from nmrf.models.NMP import MLP, Propagation, PropagationLayer
from nmrf.config import configurable


class DPN(nn.Module):
    """Disparity proposal seed extraction network.

    Args:
        cost_group (int): group number of groupwise cost volume
        num_proposals (int): number of proposals for each pixel
        feat_dim (int): dimension of backbone feature map
        context_dim (int): dimension of visual context
        prop_embed_dim (int): dimension of label seed embedding
        split_size (int): width of stripe
        prop_n_heads: head of attention
    """
    @configurable
    def __init__(self, cost_group, num_proposals, feat_dim, context_dim, num_prop_layers,
                 prop_embed_dim, mlp_ratio,  split_size, prop_n_heads, activation="gelu",
                 attn_drop=0.,  proj_drop=0., drop_path=0., dropout=0., normalize_before=False,
    ):
        super().__init__()

        # 1D convolutions sliding along the disparity dimension.
        # Intuition: high-pass filter to make the disparity modal prominent
        self.mlp = nn.Sequential(
            nn.Conv1d(cost_group, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 1, kernel_size=5, stride=1, padding=2),
        )
        self.eps = 1e-3
        self.num_proposals = num_proposals
        self.cost_group = cost_group

        # ---- label seed propagation ---- #
        # to obtain visual context
        self.proj = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, context_dim, 1, 1, 0, bias=False))

        prop_layer = PropagationLayer(prop_embed_dim, mlp_ratio=mlp_ratio, context_dim=context_dim,
                                      split_size=split_size, n_heads=prop_n_heads, activation=activation,
                                      attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path, dropout=dropout,
                                      normalize_before=normalize_before)
        prop_norm = nn.LayerNorm(prop_embed_dim)
        self.propagation = Propagation(prop_embed_dim, cost_group, prop_layer=prop_layer, num_layers=num_prop_layers,
                                       norm=prop_norm, return_intermediate=False)

        self.prop_head = MLP(prop_embed_dim, prop_embed_dim, 1, 3)

        self.apply(self._init_weights)
        nn.init.constant_(self.prop_head.layers[-1].weight.data, 0.)
        nn.init.constant_(self.prop_head.layers[-1].bias.data, 0.)

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_proposals": cfg.DPN.NUM_PROPOSALS,
            "cost_group": cfg.DPN.COST_GROUP,
            "feat_dim": cfg.BACKBONE.OUT_CHANNELS,
            "context_dim": cfg.DPN.CONTEXT_DIM,
            "num_prop_layers": cfg.NMP.NUM_PROP_LAYERS,
            "prop_embed_dim": cfg.NMP.PROP_EMBED_DIM,
            "mlp_ratio": cfg.NMP.MLP_RATIO,
            "split_size": cfg.NMP.SPLIT_SIZE,
            "prop_n_heads": cfg.NMP.PROP_N_HEADS,
            "attn_drop": cfg.NMP.ATTN_DROP,
            "proj_drop": cfg.NMP.PROJ_DROP,
            "drop_path": cfg.NMP.DROP_PATH,
            "dropout": cfg.NMP.DROPOUT,
            "normalize_before": cfg.NMP.NORMALIZE_BEFORE,
        }

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm2d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, cost_volume, fmap1_list):
        """
        cost_volume: [B,G,D,H,W]
        returns:
            cost_volume: [B*H*W,G,W] filtered cost volume
            prob: [B*H*W,D], softmax probability along the disparity dimension
            proposals: [B*H*W, num_proposals], per-pixel disparity candidate
        """
        # ---- step 1: extract disparity modals as label seeds ---- #
        bs, group, nd, ht, wd, device = *cost_volume.shape, cost_volume.device
        cost_volume = cost_volume.permute(0, 3, 4, 1, 2).reshape(bs*ht*wd, group, nd)
        cost = self.mlp(cost_volume).squeeze(-2)  # [B*H*W,D]
        prob = torch.nn.functional.softmax(cost, dim=-1)
        out = torch.nn.functional.max_pool1d(prob.unsqueeze(-2), kernel_size=3, stride=1, padding=1).squeeze(-2)
        non_local_max = (prob != out) & (prob > self.eps)

        prob_ = prob.clone().detach()
        prob_[non_local_max] = self.eps
        _, label_seeds = torch.topk(prob_, self.num_proposals, dim=-1)

        # ---- step 2: label seed propagation ---- #
        context = self.proj(fmap1_list[0])  # visual context is used in affinity computation
        context = rearrange(context, 'b c h w -> b h w c')
        memory, label_seeds = self.propagation(cost_volume, label_seeds, context)
        outputs = self.prop_head(memory).view(-1, *label_seeds.shape)
        labels = F.relu(outputs + label_seeds[None])  # candidate labels

        return cost_volume, prob, label_seeds, labels
