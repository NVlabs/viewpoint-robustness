"""
Copyright (C) 2023 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/NVlabs/viewpoint-robustness
Authors: Tzofi Klinghoffer, Jonah Philion, Wenzheng Chen, Or Litany, Zan Gojcic, Jungseock Joo, Ramesh Raskar, Sanja Fidler, Jose M. Alvarez
"""

import cv2
import json
import numpy as np
import os
import sys
import torch
from pathlib import Path
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

from tools.misc import img_transform, normalize_img, gen_dx_bx, choose_cams, parse_calibration, sample_augmentation

CARLA_CAMORDER = {
    'CAM_FRONT': 0,
    'CAM_FRONT_RIGHT': 1,
    'CAM_BACK_RIGHT': 2,
    'CAM_BACK': 3,
    'CAM_BACK_LEFT': 4,
    'CAM_FRONT_LEFT': 5,
}

class CARLADataset(object):
    """
    - dataroot: path to data directory
    - is_train: flag for training 
    - data_aug_conf: configuration
    - grid_conf: configuration for bev grid
    - ret_boxes: flag for returning the bev seg boxes from getitem() 
    - limit: max number of images
    - pitch, yaw, height: these are specified if we want to override the pitch/yaw/height specified in the info.json files (info.json contains correct extrinsics)
    """
    def __init__(self, dataroot, is_train, data_aug_conf, grid_conf, ret_boxes, limit=None, pitch=0, yaw=0, height=0):
        self.pitch = pitch
        self.yaw = yaw
        self.height = height
        self.dataroot = dataroot
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.ret_boxes = ret_boxes
        self.grid_conf = grid_conf
        self.ixes = self.get_ixes()
        if limit:
            self.ixes = self.ixes[:limit]

        # hard code this for now
        with open('./nusccalib.json', 'r') as reader:
            self.nusccalib = json.load(reader)

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        print('Carla sim:', len(self), 'is train:', self.is_train)
        print('CARLA BEV size:', len(self), '| is_train:', self.is_train)

    def get_ixes(self):
        timesteps = []
        for path in Path(self.dataroot).rglob('info.json'):
            f = str(path.parents[0])
            imgix = set(
                [int(fo.split('_')[0]) for fo in os.listdir(f) if fo != 'info.json' and fo[-4:] == '.jpg'])
            if len(imgix) == 0:
                imgix = set(
                    [int(fo.split('_')[0]) for fo in os.listdir(f) if fo != 'info.json' and fo[-4:] == '.png'])
            for img in imgix:
                timesteps.append((f, img))

        timesteps = sorted(timesteps, key=lambda x: (x[0], x[1]))
        return timesteps

    def get_image_data(self, f, fo, cams, calib, cam_adjust, use_cam_name=False):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        cal = parse_calibration(calib, width=self.data_aug_conf['W'], height=self.data_aug_conf['H'],
                                cam_adjust=cam_adjust, pitch=self.pitch, yaw=self.yaw, height_adjust=self.height)
        path = None
        for cam in cams:
            if use_cam_name:
                path = os.path.join(f, f'{fo:04}_{cam}.jpg')
            else:
                path = os.path.join(f, f'{fo:04}_{CARLA_CAMORDER[cam]:02}.jpg')
            if not os.path.isfile(path):
                path = path[:-4] + ".png"
            img = Image.open(path)

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cal[cam]['intrins'])
            rot = torch.Tensor(cal[cam]['rot'].rotation_matrix)
            tran = torch.Tensor(cal[cam]['trans'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = sample_augmentation(self.data_aug_conf, self.is_train)
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                       resize=resize,
                                                       resize_dims=resize_dims,
                                                       crop=crop,
                                                       flip=flip,
                                                       rotate=rotate,
                                                       )

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans), path)

    def get_binimg(self, gt):
        def get_box(box):
            diffw = box[:3, 1] - box[:3, 2]
            diffl = box[:3, 0] - box[:3, 1]
            diffh = box[:3, 4] - box[:3, 0]

            center = (box[:3, 4] + box[:3, 2]) / 2
            # carla flips y axis
            center[1] = -center[1]

            dims = [np.linalg.norm(diffw), np.linalg.norm(diffl), np.linalg.norm(diffh)]

            rot = np.zeros((3, 3))
            rot[:, 1] = diffw / dims[0]
            rot[:, 0] = diffl / dims[1]
            rot[:, 2] = diffh / dims[2]

            quat = Quaternion(matrix=rot)
            # again, carla flips y axis
            newquat = Quaternion(quat.w, -quat.x, quat.y, -quat.z)

            nbox = Box(center, dims, newquat)
            return nbox

        boxes = [get_box(box) for box in gt]

        img = np.zeros((int(self.nx[0]), int(self.nx[1])))
        for nbox in boxes:
            pts = nbox.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0), boxes

    def __len__(self):
        return len(self.ixes)

    def __getitem__(self, index):
        f, fo = self.ixes[index]

        cams = choose_cams(self.is_train, self.data_aug_conf)

        with open(os.path.join(f, 'info.json'), 'r') as reader:
            gt = json.load(reader)
        if not 'cam_adjust' in gt:
            gt['cam_adjust'] = {k: {'fov': 0.0, 'yaw': 0.0} for k in CARLA_CAMORDER}

        imgs, rots, trans, intrins, post_rots, post_trans, path = self.get_image_data(f, fo, cams, self.nusccalib[gt['scene_calib']],
                                                                                gt['cam_adjust'],
                                                                                use_cam_name=False)
        binimg, boxes = self.get_binimg(np.array(gt['boxes'][fo]))

        if self.ret_boxes:
            if len(boxes) > 0:
                pts = torch.cat([torch.Tensor(box.bottom_corners()[:3]) for box in boxes], 1)
            else:
                return imgs, rots, trans, intrins, post_rots, post_trans, binimg, binimg
            return imgs, rots, trans, intrins, post_rots, post_trans, pts, binimg

        return imgs, rots, trans, intrins, post_rots, post_trans, binimg, path

def worker_rnd_init(x):
    np.random.seed(13 + x)

def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, pitch=0, yaw=0, height=0,limit=5000):
    traindata = CARLADataset(dataroot, True, data_aug_conf, grid_conf, ret_boxes=False, pitch=pitch, yaw=yaw, height=height)
    val_root = "/".join(dataroot.split("/")[:-1]) + "/town05"
    valdata = CARLADataset(val_root, True, data_aug_conf, grid_conf, ret_boxes=False, limit=limit, pitch=pitch, yaw=yaw, height=height)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)
    return trainloader, valloader

if __name__ == "__main__":
    H=512
    W=512
    resize_lim=(0.339, 0.703)
    final_dim=(128, 352)
    bot_pct_lim=(0.0, 0.22)
    rot_lim=(-5.4, 5.4)
    rand_flip=True
    ncams=1
    xbound=[-50.0, 50.0, 0.5]
    ybound=[-50.0, 50.0, 0.5]
    zbound=[-10.0, 10.0, 20.0]
    dbound=[4.0, 45.0, 1.0]
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT'],
                    'Ncams': ncams,
                }

    data = CARLADataset(os.path.join(sys.argv[1], "home/carla/town03"),
                     True,
                     data_aug_conf,
                     grid_conf,
                     ret_boxes=False)
