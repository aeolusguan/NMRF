import itertools
import torch
from torch import nn


def bf_match(outputs, targets, num=4):
    """ This function computes an assignment between the targets and the predictions of the network.

    The targets include null (=0) disparity. We do a 1-to-1 matching between non-null targets and best predictions,
    while the others are un-matched (and thus not propagate gradients).

    outputs: This is a tensor of dim [batch_size*H*W, num_queries]
    targets: This is a tensor of dim [batch_size*H*W, 4] containing the target disparities

    Returns:
        index tensor of dim [batch_size*H*W, 4], where ith element of each row denotes the index of matched proposal
            with ith ground truth.
    """
    # concat the target disparities
    tgt_disp = targets.reshape(-1, num, 1)

    out_disp = outputs.unsqueeze(-1)
    num_seed = out_disp.shape[1]
    assert num_seed >= num

    # compute the absolute difference between disparities
    cost_disp = torch.cdist(tgt_disp, out_disp, p=1)  # [batch_size*H*W, num, num_queries]
    # reset the cost between null and any prediction to 1e5, such that their existence will not affect the matching
    tgt_disp = tgt_disp.reshape(-1, num)
    cost_disp[tgt_disp == 0, :] = 1e5
    cost_disp_flatten = torch.flatten(cost_disp, 1)
    disp_error, _ = torch.min(cost_disp_flatten, dim=1)

    perms = list(itertools.permutations(range(num_seed), num))
    total_cost = []
    for perm in perms:
        total_cost.append(cost_disp[:, range(num), perm].sum(dim=-1, keepdims=True))
    total_cost = torch.cat(total_cost, dim=-1)
    _, indices = torch.min(total_cost, dim=-1)
    perms = torch.tensor(list(perms)).to(outputs.device)
    return perms[indices, :], disp_error


class NearestMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    The targets include null (=0) disparity. We do a many-to-one matching and identify the best predictions
    among the ones assigned to a same target.
    """

    def __init__(self, cost_class: float = 1, cost_disp: float = 1):
        """Create the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_disp: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_disp = cost_disp
        assert cost_class != 0 or cost_disp != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                "pred_logits": Tensor of dim [batch_size, num_queries] with the classification logits
                "pred_disp": Tensor of dim [batch_size, num_queries] with the predicted disparity

            targets: This is a tensor of target disparities, of dim [batch_size, num_targets]

        Returns:
            A Tensor of dim [batch_size, num_queries], containing the indices of the corresponding matched targets
            A Tensor of dim [batch_size, num_queries], the classification targets of predictions
        """
        pred_logits = outputs['pred_logits']
        pred_disp = outputs['pred_disp']
        tgt_disp = targets.clone()
        tgt_disp[tgt_disp == 0] = 1e6  # avoid predictions matched with null target
        dist = torch.abs(pred_disp[:, :, None] - tgt_disp[:, None, :])
        cost_disp, indices = torch.min(dist, dim=-1, keepdim=False)
        cost_class = -pred_logits.sigmoid()

        C = self.cost_class * cost_class + self.cost_disp * cost_disp
        # NMS the predictions that are matched with the same target
        idx = C.argsort(dim=-1, descending=False)
        tmp_indices = torch.gather(indices, dim=-1, index=idx)
        N = pred_logits.shape[-1]
        labels = torch.ones_like(pred_logits)
        for i in range(N-1):
            suppress = tmp_indices[:, i:i+1] == tmp_indices[:, i+1:]
            labels[:, i+1:][suppress] = 0
        labels = torch.ones_like(labels).scatter_(dim=-1, index=idx, src=labels)
        return indices, labels