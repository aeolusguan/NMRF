import logging
import os

import numpy as np
import torch
import cv2

from nmrf.utils import frame_utils
from nmrf.data import datasets


def disp_cross_check(dispL, dispR):
    H, W = dispL.shape
    gridX = torch.arange(0, W, dtype=torch.float32).view(1, W).repeat(H, 1)
    gridY = torch.arange(0, H, dtype=torch.float32).view(H, 1).repeat(1, W)
    # forward warp
    gridX_forward = gridX - dispL
    valid = gridX_forward >= 0  # the mask for inside right image region
    # backward warp
    xs = 2 * gridX_forward / (W - 1) - 1
    ys = 2 * gridY / (H - 1) - 1
    grid = torch.stack((xs, ys), dim=-1)[None]
    dispR = dispR.reshape(1, 1, H, W)
    disp = torch.nn.functional.grid_sample(dispR, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    gridX_backward = gridX_forward + disp[0, 0, :, :]
    valid = valid & (torch.abs(gridX_backward - gridX) < 1)
    return valid


def gen_occ_map_sceneflow():
    dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass')
    logging.info(f"Dataset length {len(dataset.disparity_list)} from SceneFlow")
    for fn in dataset.disparity_list:
        dispL = frame_utils.readPFM(fn)
        dispL = torch.from_numpy(np.array(dispL).astype(np.float32))
        fnR = fn.replace('left', 'right')
        dispR = frame_utils.readPFM(fnR)
        dispR = torch.from_numpy(np.array(dispR).astype(np.float32))
        nocc = disp_cross_check(dispL, dispR)
        mask_path = os.path.join(os.path.dirname(fn), os.path.splitext(os.path.basename(fn))[0] + '_nocc.png')
        cv2.imwrite(mask_path, nocc.numpy() * 255)


if __name__ == '__main__':
    gen_occ_map_sceneflow()
