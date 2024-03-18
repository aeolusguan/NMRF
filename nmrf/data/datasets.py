# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import copy
from pathlib import Path
from glob import glob
import os.path as osp
import cv2

from nmrf.utils import frame_utils
from nmrf.utils import dist_utils as comm
from nmrf.utils import misc
from nmrf.utils import evaluation
from .transforms import FlowAugmentor, SparseFlowAugmentor


# read all lines in a file
def read_all_lines(filename):
    with open(filename) as fp:
        lines = [line.rstrip() for line in fp.readlines()]
    return lines


class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if self.sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.occ_map_list = []
        self.extra_info = []

    def __getitem__(self, index):

        sample = {}
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            sample['img1'] = torch.from_numpy(img1).permute(2, 0, 1).float()
            sample['img2'] = torch.from_numpy(img2).permute(2, 0, 1).float()
            sample['meta'] = self.extra_info[index]
            return sample

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            initial_seed = torch.initial_seed() % 2**31
            if worker_info is not None:
                misc.seed_all_rng(initial_seed + worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disparity_list[index])
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        super_pixel_label = self.image_list[index][0][:-len('.png')] + "_lsc_lbl.png"
        if not os.path.exists(super_pixel_label):
            img = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
            lsc = cv2.ximgproc.createSuperpixelLSC(img, region_size=10, ratio=0.075)
            lsc.iterate(20)
            label = lsc.getLabels()
            cv2.imwrite(super_pixel_label, label.astype(np.uint16))
        super_pixel_label = frame_utils.read_super_pixel_label(super_pixel_label)

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        disp = np.array(disp).astype(np.float32)

        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        try:
            occlusion_map = frame_utils.readOcclusionMap(self.occ_map_list[index])[..., 0] < 128
        except Exception as e:
            occlusion_map = np.zeros_like(disp, dtype=bool)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, super_pixel_label, occlusion_map, occlusion_map_2, valid = self.augmentor(img1, img2, flow, super_pixel_label, occlusion_map, valid)
            else:
                img1, img2, flow, super_pixel_label, occlusion_map, occlusion_map_2 = self.augmentor(img1, img2, flow, super_pixel_label, occlusion_map)
        else:
            occlusion_map_2 = np.zeros(img1.shape[:2], dtype=bool)

        sample['img1'] = torch.from_numpy(img1).permute(2, 0, 1).float()
        sample['img2'] = torch.from_numpy(img2).permute(2, 0, 1).float()
        sample['disp'] = torch.from_numpy(flow).permute(2, 0, 1).float()[0]

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = sample['disp'] < 512

        sample['super_pixel_label'] = torch.from_numpy(super_pixel_label)
        sample['occlusion_map'] = torch.from_numpy(occlusion_map)
        sample['occlusion_map_2'] = torch.from_numpy(occlusion_map_2)
        sample['valid'] = valid

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        return sample

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.occ_map_list = v * copy_of_self.occ_map_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self

    def __len__(self):
        return len(self.image_list)


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/SceneFlow', dstype='frames_finalpass', things_test=False):
        super(SceneFlowDatasets, self).__init__(aug_params)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa()
            self._add_driving()

    def _add_things(self, split='TRAIN'):
        """ Add Flythings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'FlyingThings3D')
        left_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/left/*.png')) )
        right_images = [ im.replace('left', 'right') for im in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]
        occ_map_images = [ im.replace('.pfm', '_nocc.png') for im in disparity_images ]

        for img1, img2, disp, occ_map in zip(left_images, right_images, disparity_images, occ_map_images):
            if img1.endswith('_lsc_lbl.png'):
                continue
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
            if split == 'TRAIN':
                self.occ_map_list += [ occ_map ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self):
        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Monkaa')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]
        occ_map_images = [ im.replace('.pfm', '_nocc.png') for im in disparity_images ]

        for img1, img2, disp, occ_map in zip(left_images, right_images, disparity_images, occ_map_images):
            if img1.endswith('_lsc_lbl.png'):
                continue
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
            self.occ_map_list += [ occ_map ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")

    def _add_driving(self):
        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Driving')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]
        occ_map_images = [ im.replace('.pfm', '_nocc.png') for im in disparity_images ]

        for img1, img2, disp, occ_map in zip(left_images, right_images, disparity_images, occ_map_images):
            if img1.endswith('_lsc_lbl.png'):
                continue
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
            self.occ_map_list += [ occ_map ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


class Carla(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/Carla'):
        super(Carla, self).__init__(aug_params, reader=frame_utils.readDispCarla)
        self.root = root

        left_images = sorted(glob(osp.join(root, f'*/generated/images_rgb/*_0.png')))
        right_images = sorted(glob(osp.join(root, f'*/generated/images_rgb/*_1.png')))
        disparity_images = sorted(glob(osp.join(root, f'*/generated/images_depth/*_20.png')))

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list)} From Carla")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/ETH3D', split='training'):
        super(ETH3D, self).__init__(aug_params, sparse=True)

        image1_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im0.png')) )
        image2_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im1.png')) )
        disp_list = sorted( glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm')) ) if split == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')]*len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]


class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/SintelStereo'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo)

        image1_list = sorted( glob(osp.join(root, 'training/*_left/*/frame_*.png')) )
        image2_list = sorted( glob(osp.join(root, 'training/*_right/*/frame_*.png')) )
        disp_list = sorted( glob(osp.join(root, 'training/disparities/*/frame_*.png')) ) * 2

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]


class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/FallingThings'):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)

        with open(os.path.join(root, 'filenames.txt'), 'r') as f:
            filenames = sorted(f.read().splitlines())

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('left.jpg', 'right.jpg')) for e in filenames]
        disp_list = [osp.join(root, e.replace('left.jpg', 'left.depth.png')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]


class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='datasets', keywords=[]):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)

        with open(os.path.join(root, 'tartanair_filenames.txt'), 'r') as f:
            filenames = sorted(list(filter(lambda s: 'seasonsforest_winter/Easy' not in s, f.read().splitlines())))
            for kw in keywords:
                filenames = sorted(list(filter(lambda s: kw in s.lower(), filenames)))

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('_left', '_right')) for e in filenames]
        disp_list = [osp.join(root, e.replace('image_left', 'depth_left').replace('left.png', 'left_depth.npy')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]


class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/KITTI', split='training', image_set='kitti_mix'):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)

        if split == 'testing':
            self.is_test = True
            if image_set == 'kitti_2012':
                root = osp.join(root, 'KITTI_2012')
                images1 = sorted(glob(osp.join(root, 'testing', 'colored_0/*_10.png')))
                images2 = sorted(glob(osp.join(root, 'testing', 'colored_1/*_10.png')))
            elif image_set == 'kitti_2015':
                root = osp.join(root, 'KITTI_2015')
                images1 = sorted(glob(osp.join(root, 'testing', 'image_2/*_10.png')))
                images2 = sorted(glob(osp.join(root, 'testing', 'image_3/*_10.png')))
            else:
                raise ValueError("Unknown dataset for test: '{}'".format(image_set))
            for img1, img2 in zip(images1, images2):
                frame_id = img1.split('/')[-1]
                self.extra_info += [ frame_id ]
                self.image_list += [ [img1, img2] ]

        else:
            kitti_dict = {
                'kitti_mix_2012_train': 'filenames/KITTI_mix_2012_train.txt',
                'kitti_mix_2015_train': 'filenames/KITTI_mix_2015_train.txt',
                'kitti_2012_val': 'filenames/KITTI_2012_val.txt',
                'kitti_2015_val': 'filenames/KITTI_2015_val.txt',
                'kitti_mix': 'filenames/KITTI_mix.txt',
                'kitti_2015_train': 'filenames/KITTI_2015_train.txt',
                'kitti_2015_trainval': 'filenames/KITTI_2015_trainval.txt',
                'kitti_2012_train': 'filenames/KITTI_2012_train.txt',
                'kitti_2012_trainval': 'filenames/KITTI_2012_trainval.txt',
            }

            assert image_set in kitti_dict.keys()
            data_filename = kitti_dict[image_set]

            self._root_12 = os.path.join(root, 'KITTI_2012')
            self._root_15 = os.path.join(root, 'KITTI_2015')

            self.load_path(data_filename)

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        for line in splits:
            left_name = line[0].split('/')[1]
            if left_name.startswith('image'):
                root = self._root_15
            else:
                root = self._root_12
            img1 = os.path.join(root, line[0])
            img2 = os.path.join(root, line[1])
            self.image_list += [ [img1, img2] ]
            if len(line) > 2:
                disp = os.path.join(root, line[2])
                self.disparity_list += [ disp ]
            frame_id = img1.split('/')[-1]
            self.extra_info += [ frame_id ]


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/Middlebury', split='F'):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury)
        assert os.path.exists(root)
        assert split in ["F", "H", "Q", "2014"]
        if split == "2014": # datasets/Middlebury/2014/Pipes-perfect/im0.png
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E","L",""]:
                    self.image_list += [ [str(scene / "im0.png"), str(scene / f"im1{s}.png")] ]
                    self.disparity_list += [ str(scene / "disp0.pfm") ]
        else:
            lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/trainingF/*"))))
            lines = list(filter(lambda p: any(s in p.split('/') for s in Path(os.path.join(root, "MiddEval3/official_train.txt")).read_text().splitlines()), lines))
            image1_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im0.png') for name in lines])
            image2_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im1.png') for name in lines])
            disp_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/disp0GT.pfm') for name in lines])
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]


def build_train_loader(cfg):
    """ Create the data loader for the corresponding training set """
    crop_size = cfg.DATASETS.CROP_SIZE
    spatial_scale = cfg.DATASETS.SPATIAL_SCALE
    yjitter = cfg.DATASETS.YJITTER
    aug_params = {'crop_size': list(crop_size), 'min_scale': spatial_scale[0], 'max_scale': spatial_scale[1], 'do_flip': False, 'yjitter': yjitter}
    if cfg.DATASETS.SATURATION_RANGE is not None:
        aug_params["saturation_range"] = cfg.DATASETS.SATURATION_RANGE
    if cfg.DATASETS.IMG_GAMMA is not None:
        aug_params["gamma"] = cfg.DATASETS.IMG_GAMMA
    if cfg.DATASETS.DO_FLIP is not None:
        aug_params["do_flip"] = cfg.DATASETS.DO_FLIP

    train_dataset = None
    logger = logging.getLogger(__name__)
    for dataset_name in cfg.DATASETS.TRAIN:
        if dataset_name.startswith("middlebury_"):
            new_dataset = Middlebury(aug_params, split=dataset_name.replace('middlebury_', ''))
        elif dataset_name == 'sceneflow':
            new_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
            logger.info(f"Adding {len(new_dataset)} samples from SceneFlow")
        elif 'kitti' in dataset_name:
            new_dataset = KITTI(aug_params, image_set=dataset_name)
            logger.info(f"Adding {len(new_dataset)} samples from KITTI")
        elif dataset_name == 'sintel_stereo':
            new_dataset = SintelStereo(aug_params) * 140
            logger.info(f"Adding {len(new_dataset)} samples from Sintel Stereo")
        elif dataset_name == 'falling_things':
            new_dataset = FallingThings(aug_params) * 5
            logger.info(f"Adding {len(new_dataset)} samples from FallingThings")
        elif dataset_name.startswith('tartan_air'):
            new_dataset = TartanAir(aug_params, keywords=dataset_name.split('_')[2:])
            logger.info(f"Adding {len(new_dataset)} samples from Tartain Air")
        elif dataset_name == "carla":
            new_dataset = Carla(aug_params)
            logger.info(f"Adding {len(new_dataset)} samples from Carla")
        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    world_size = comm.get_world_size()
    total_batch_size = cfg.SOLVER.IMS_PER_BATCH
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({})".format(
        total_batch_size, world_size
    )
    batch_size = cfg.SOLVER.IMS_PER_BATCH // world_size

    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=comm.get_world_size(),
            rank=comm.get_rank())
        shuffle = False
    else:
        train_sampler = None
        shuffle = False
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=cfg.DATALOADER.NUM_WORKERS,
                                               pin_memory=True, drop_last=True,
                                               sampler=train_sampler)

    return train_loader, train_sampler


def build_val_loader(cfg, dataset_name):
    logger = logging.getLogger(__name__)
    if dataset_name == 'things':
        val_dataset = SceneFlowDatasets(dstype='frames_finalpass', things_test=True)
        logger.info('Number of validation image pairs: %d' % len(val_dataset))
    elif 'kitti' in dataset_name:
        # perform validation using the KITTI (train) split
        val_dataset = KITTI(image_set=dataset_name)
        logger.info('Number of validation image pairs: %d' % len(val_dataset))
    elif dataset_name == 'eth3d':
        val_dataset = ETH3D(split='training')
        logger.info('Number of validation image pairs: %d' % len(val_dataset))
    elif dataset_name.startswith("middlebury_"):
        val_dataset = Middlebury(split=dataset_name.replace('middlebury_', ''))
        logger.info('Number of validation image pairs: %d' % len(val_dataset))
    else:
        raise ValueError("Unknown dataset: '{}'".format(dataset_name))

    world_size = comm.get_world_size()
    if world_size > 1:
        val_sampler = evaluation.InferenceSampler(len(val_dataset))
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                                             pin_memory=True, drop_last=False,
                                             sampler=val_sampler)
    return val_loader