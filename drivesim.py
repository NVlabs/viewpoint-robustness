"""
Copyright (C) 2023 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/NVlabs/viewpoint-robustness
Authors: Tzofi Klinghoffer, Jonah Philion, Wenzheng Chen, Or Litany, Zan Gojcic, Jungseock Joo, Ramesh Raskar, Sanja Fidler, Jose M. Alvarez
"""

import cv2
import json
import numpy as np
import os
import time
import torch
import torchvision
from numpy.polynomial.polynomial import Polynomial
from pathlib import Path
from PIL import Image

from tools.augmentations import StrongAug, GeometricAug
from tools.common import get_view_matrix
from tools.data_tools import parse, camera_intrinsic_parameters, sensor_to_rig, compute_ftheta_fov, compute_cuboid_corners, \
        degree_to_radian, parse_rig_sensors_from_file, parse_drivesim
from tools.misc import img_transform, normalize_img, gen_dx_bx, sample_augmentation
from tools.sensor_models import FThetaCamera, IdealPinholeCamera, rectify
from tools.transforms import Sample

"""
 Maps datasets (e.g. fov120_0, ... fov120_11) to extrinsics (pitch, height, x; x is depth)
 pitch units are degrees, height and x are in meters
 positive pitch ==> camera looks up
 positive x ==> camera moves forward
 positive height ==> camera moves up
"""
DATASETS = {
    '0': [0,0,0],
    '2': [0,0,1.5],
    '3': [0,0.2,0],
    '4': [0,0.4,0],
    '5': [0,0.6,0],
    '6': [0,0.8,0],
    '7': [5,0,0],
    '8': [10,0,0],
    '9': [10,0.6,0],
    '10': [-5,0,0],
    '11': [-10,0,0]
}

CAMORDER = {
    'CAM_FRONT': 0,
    'CAM_FRONT_RIGHT': 1,
    'CAM_BACK_RIGHT': 2,
    'CAM_BACK': 3,
    'CAM_BACK_LEFT': 4,
    'CAM_FRONT_LEFT': 5,
}

class DRIVESimDataset(object):
    """
    Logic for loading DRIVE Sim data. Assumes data has first been rectified from FTheta to Pinhole model.

    Args:
    - dataroot: path to data directory
    - im_path: path to subdirectory containing images of interest
    - sessions: session id (stored as list)
    - data_aug_conf: data configuration
    - grid_conf: bev segmentation configuration
    - ret_boxes: boolean specifying whether boxes are returned in getitem()
    - pitch: pitch of data
    - height: height of data
    - x: depth of data
    - camname: camera name (str)
    - rectified_fov: field of view (either to rectify to if rectifying or of the rectified data)
    - start: start index to load data
    - stop: stop index to load data (e.g. will load images[start:stop]
    - viz: boolean specifying whether to visualize data when getitem() is called, will save images locally
    """
    def __init__(self,
                 dataroot,
                 im_path,
                 sessions,
                 data_aug_conf,
                 grid_conf,
                 ret_boxes,
                 pitch=0,
                 height=0,
                 x=0,
                 camname="camera:front:wide:120fov",
                 rectified_fov=50,
                 start=0,
                 stop=None,
                 viz=False,
                 rectify_data=False):
        self.camname = camname
        self.rectified_fov = rectified_fov
        self.path = im_path
        self.camera = im_path.split("/")[-2]
        self.viz = viz
        self.rectify_data = rectify_data

        self.pitch = pitch
        self.height = height
        self.x = x

        self.dataroot = dataroot
        self.sessions = sessions
        self.data_aug_conf = data_aug_conf
        self.ret_boxes = ret_boxes
        self.grid_conf = grid_conf
        self.ixes, self.data = self.get_ixes()
        if stop is None:
            self.ixes = self.ixes[start:]
        else:
            self.ixes = self.ixes[start:stop]

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        print('Dataset:', len(self))

        bev = {'h': 200, 'w': 200, 'h_meters': 100, 'w_meters': 100, 'offset': 0.0}
        self.view = get_view_matrix(**bev)

    def get_ixes(self):
        train = []
        val = []
        test = []
        samples = []
        data = {}
        start = time.time()
        frame_count = 0
        bbox_count = 0
        for i, f in enumerate(self.sessions):
            sqlite = None
            ignore = []
            fcount = 0
            bcount = 0
            sqlite = os.path.join(self.dataroot, "dataset.sqlite") 
            self.session_rig = os.path.join(self.dataroot, "sql_template", "session_rigs", f) + ".json"
            data, fcount, bcount = self.parse_drivesim(data, os.path.join(self.dataroot, "sql_template", "session_rigs", f) + ".json", \
                                    os.path.join(self.dataroot, "sql_template", "features", f, self.camera), \
                                    self.path)
            imgix = [(self.path, sqlite, fo) for fo in os.listdir(self.path) \
                    if fo[-5:] == '.jpeg' and os.path.join(self.path,fo) not in ignore]
            samples.extend(imgix)
            frame_count += fcount
            bbox_count += bcount

        print("Frames: {}, Average boxes per frame: {}".format(frame_count, float(bbox_count)/float(frame_count)))
        return samples, data

    def get_image_data(self, path, intrin, rot, tran, extrin):
        cams = ["CAM_FRONT"]
        augment = 'none'
        xform = {
            'none': [],
            'strong': [StrongAug()],
            'geometric': [StrongAug(), GeometricAug()],
        }[augment] + [torchvision.transforms.ToTensor()]

        self.img_transform = torchvision.transforms.Compose(xform)

        imgs = []
        rots = []
        trans = []
        intrins = []
        extrins = []
        cam_rig = []
        cam_channel = []

        for cam in cams:
            image = Image.open(path)
            if image.size != (1924, 1084):
                image = image.resize((1924,1084))

            intrin = torch.Tensor(intrin)
            rot = torch.Tensor(rot)
            tran = torch.Tensor(tran)
            extrin = torch.Tensor(extrin)

            h = self.data_aug_conf['H']
            w = self.data_aug_conf['W']
            top_crop=46
            h_resize = h
            w_resize = w
            image_new = image.resize((w_resize, h_resize), resample=Image.Resampling.BILINEAR)
            I = np.float32(intrin)
            I[0, 0] *= w_resize / image.width
            I[0, 2] *= w_resize / image.width
            I[1, 1] *= h_resize / image.height
            I[1, 2] *= h_resize / image.height

            img = self.img_transform(image_new)
            imgs.append(img)

            intrins.append(torch.tensor(I))
            extrins.append(extrin.tolist())
            rots.append(rot)
            trans.append(tran)
            cam_rig.append(CAMORDER[cam])
            cam_channel.append(cam)

        return {
            'cam_ids': torch.LongTensor(cam_rig),
            'cam_channels': cam_channel,
            'intrinsics': torch.stack(intrins,0),
            'extrinsics': torch.tensor(np.float32(extrins)),
            'rots': rots,
            'trans': trans,
            'image': torch.stack(imgs,0),
        }

    def get_binimg(self, boxes, rig2sensor, ph_model):
        thickness = 1
        img = np.zeros((int(self.nx[0]), int(self.nx[1])))
        boxz = []
        points = []
        for i, cuboid3d in enumerate(boxes):
            cuboid3d_array = np.array(
                    [
                        cuboid3d.center.x,
                        cuboid3d.center.y,
                        cuboid3d.center.z,
                        cuboid3d.dimension.x,
                        cuboid3d.dimension.y,
                        cuboid3d.dimension.z,
                        cuboid3d.orientation.yaw,
                        cuboid3d.orientation.pitch,
                        cuboid3d.orientation.roll,
                    ]
                )
            cuboid3d_array[6:9] = degree_to_radian(cuboid3d_array[6:9])
            cuboid3d_corners = compute_cuboid_corners(
                cuboid3d_array, use_polar=False, use_mrp=False)

            points.append(torch.tensor(cuboid3d_corners[4]))
            points.append(torch.tensor(cuboid3d_corners[5]))
            points.append(torch.tensor(cuboid3d_corners[6]))
            points.append(torch.tensor(cuboid3d_corners[7]))

            corners = cuboid3d_corners[4:8].T

            pts = corners[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0), torch.stack(points, -1).float()

    def rectify_drivesim(self, key, rig_path):
        DOWNSAMPLE = 4.0
        rig = parse_rig_sensors_from_file(rig_path)[self.camname]
        sensor2rig = sensor_to_rig(rig)
        fisheye_intrins = camera_intrinsic_parameters(rig)
        size = np.array([float(rig['properties']['width']), float(rig['properties']['height'])])
        bwpoly = fisheye_intrins[4:]
        fov_x, fov_y, _ = compute_ftheta_fov(fisheye_intrins)
        fx = (fisheye_intrins[2] / (2.0 * np.tan(float(fov_x / 2.0 )))) / DOWNSAMPLE
        fy = (fisheye_intrins[3] / (2.0 * np.tan(float(fov_y / 2.0 )))) / DOWNSAMPLE

        fx = ((float(fisheye_intrins[3]) / DOWNSAMPLE) / 2.0 / np.tan(float(fov_x / 2.0 )))
        fy = ((float(fisheye_intrins[2]) / DOWNSAMPLE) / 2.0 / np.tan(float(fov_y / 2.0 )))

        focal = (float(size[0]) / DOWNSAMPLE) / 2.0 / np.tan(np.radians(120.0/2.0))
        cx = fisheye_intrins[0] / DOWNSAMPLE
        cy = fisheye_intrins[1] / DOWNSAMPLE
        intrins = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

        properties = rig['properties']
        bw_poly = Polynomial(fisheye_intrins[4:])
        downsample_pixel_map = np.polynomial.Polynomial([0.0, DOWNSAMPLE])
        new_bw_poly = bw_poly(downsample_pixel_map)
        ftheta_model = FThetaCamera(
            cx=float(properties['cx']) / DOWNSAMPLE,
            cy=float(properties['cy']) / DOWNSAMPLE,
            width=size[0] / DOWNSAMPLE,
            height=size[1] / DOWNSAMPLE,
            bw_poly=new_bw_poly.convert().coef,
        )

        target_fov = np.radians(self.rectified_fov)
        target_foc = (ftheta_model.width / 2) / np.tan(target_fov / 2)
        ph_model = IdealPinholeCamera(
            fov_x_deg=None,
            fov_y_deg=None,
            width=ftheta_model.width,
            height=ftheta_model.height,
            f_x=target_foc,
            f_y=target_foc,
        )

        img = cv2.imread(key)
        img = rectify(img, ftheta_model, ph_model)

        fname = key.split("/")[-1]
        #save_path = "/".join(key.split("/")[:-1]).replace("rgb_jpeg", "rgb_jpeg_rectified")
        save_path = "/".join(key.split("/")[:-1]).replace("rgb_half_jpeg-100-xavierisp", "rgb_jpeg_rectified2")
        if not os.path.exists(save_path):
            path = Path(save_path)
            path.mkdir(parents=True)
        cv2.imwrite(os.path.join(save_path, fname), img)

    def parse_drivesim(self, data, rig_path, json_path, img_path):
        DOWNSAMPLE = 2.0 # images are 4x smaller than specified in the json, but then upsampled 2x in get_image_data()
        rig = parse_rig_sensors_from_file(rig_path)[self.camname]

        rig["nominalSensor2Rig_FLU"]["roll-pitch-yaw"][1] += self.pitch
        rig["nominalSensor2Rig_FLU"]["t"][2] += self.height
        rig["nominalSensor2Rig_FLU"]["t"][0] += self.x

        sensor2rig, _ = sensor_to_rig(rig)
        print("DriveSIM x {}, y {}, z {}".format(sensor2rig[0,3],sensor2rig[1,3],sensor2rig[2,3]))
        fisheye_intrins = camera_intrinsic_parameters(rig)
        fov_x, fov_y, _ = compute_ftheta_fov(fisheye_intrins)
        size = np.array([float(rig['properties']['width']), float(rig['properties']['height'])])

        target_fov = np.radians(self.rectified_fov)
        target_foc = ((size[0]/DOWNSAMPLE) / 2) / np.tan(target_fov / 2)
        cx = fisheye_intrins[0] / DOWNSAMPLE
        cy = fisheye_intrins[1] / DOWNSAMPLE
        intrins = np.array([[target_foc,0,cx],[0,target_foc,cy],[0,0,1]])

        ph_model = IdealPinholeCamera(
            fov_x_deg=None,
            fov_y_deg=None,
            width=size[0] / DOWNSAMPLE,
            height=size[1] / DOWNSAMPLE,
            f_x=target_foc,
            f_y=target_foc,
        )

        frame_count = 0
        bbox_count = 0

        for f in os.listdir(json_path):
            key = os.path.join(img_path, f[:-5]) + ".jpeg"
            if self.rectify_data:
                self.rectify_drivesim(key, rig_path)
                continue
            with open(os.path.join(json_path, f)) as fp:
                d = json.load(fp)
            boxes = []
            for item in d:
                if item["label_family"] == "SHAPE2D" and item["label_name"] == "automobile":
                    cuboid3d, ratio = parse_drivesim(json.loads(item["data"]))
                    if float(ratio) < 0.5:
                        boxes.append(cuboid3d)

            if len(boxes) > 0:
                data[key] = {"boxes": boxes,
                             "intrins": intrins,
                             "sensor2rig": sensor2rig,
                             "fisheye_intrins": fisheye_intrins,
                             "size": size,
                             "ph_model": ph_model} 
                frame_count += 1
                bbox_count += len(boxes)

        return data, frame_count, bbox_count

    def pinhole_project(self, points, img, intrin, sensor2rig, ph_model):
        intrin = intrin.detach().cpu().numpy()
        intrin = np.array(intrin, dtype=np.float64)
        rig2sensor = np.linalg.inv(sensor2rig)
        points = torch.swapaxes(points, 0, 1)
        for point in points:
            point = np.array([point[0],point[1],point[2],1]).T
            point = np.dot(rig2sensor, point)[:-1]
            cam_pixels = ph_model.ray2pixel(point.T)[0][0]
            x = int(cam_pixels[0])
            y = int(cam_pixels[1])
            img = cv2.circle(img, (x,y), radius=2, color=(0, 0, 255), thickness=2)
        return img

    def ftheta_project(self, points, img, sensor2rig):
        ftheta_model = FThetaCamera.from_rig(self.session_rig, self.camname)
        points = torch.swapaxes(points, 0, 1)
        for i, point in enumerate(points):
            rig2sensor = np.linalg.inv(sensor2rig)
            point = np.array([point[0],point[1],point[2],1]).T
            point = np.dot(rig2sensor, point)[:-1]
            cam_pixels = ftheta_model.ray2pixel(point.T)[0] / 4.0
            x = int(cam_pixels[0])
            y = int(cam_pixels[1])
            img = cv2.circle(img, (x,y), radius=2, color=(0, 0, 255), thickness=2)
        return img

    def is_point_in_fov(self, point, rig2sensor, ph_model):
        point = np.array([point[0], point[1], point[2], 1]).T
        cam_point = np.dot(rig2sensor, point)[:-1]
        cam_pixels = ph_model.ray2pixel(cam_point.T)[0][0]
        x = int(cam_pixels[0])
        y = int(cam_pixels[1])
        if x >= 0 and x <= ph_model.width and \
                y >= 0 and y <= ph_model.height:
            return True
        return False

    def is_box_in_fov(self, cuboid3d, rig2sensor, ph_model):
        cuboid3d_array = np.array(
                [
                    cuboid3d.center.x,
                    cuboid3d.center.y,
                    cuboid3d.center.z,
                    cuboid3d.dimension.x,
                    cuboid3d.dimension.y,
                    cuboid3d.dimension.z,
                    cuboid3d.orientation.yaw,
                    cuboid3d.orientation.pitch,
                    cuboid3d.orientation.roll,
                ]
            )
        cuboid3d_array[6:9] = degree_to_radian(cuboid3d_array[6:9])
        cuboid3d_corners = compute_cuboid_corners(
            cuboid3d_array, use_polar=False, use_mrp=False)#[:8]

        inside = 0
        for corner in cuboid3d_corners:
            inside += int(self.is_point_in_fov(corner, rig2sensor, ph_model))

        return bool(inside)

    def __len__(self):
        return len(self.ixes)

    def __getitem__(self, index):
        ipath, sqlite, fname = self.ixes[index]

        data = self.data[os.path.join(ipath,fname)]
        boxes = data["boxes"]
        intrins = data["intrins"]
        sensor2rig = data["sensor2rig"]
        ph_model = data["ph_model"]

        sample = self.get_image_data(os.path.join(ipath, fname), intrins, sensor2rig[:3,:3], sensor2rig[:3,3], sensor2rig)
        binimg, points = self.get_binimg(boxes, np.linalg.inv(sensor2rig), ph_model)

        # Visualizations (save image with projected points and ground truth bev segmentation map)
        if self.viz:
            im = cv2.imread(os.path.join(ipath,fname))
            im = self.ftheta_project(points, im, sensor2rig)
            idx = fname.split(".")[0]
            cv2.imwrite("viz_{}.png".format(idx.zfill(5)), im)
            cv2.imwrite("gt_{}.png".format(idx.zfill(5)), np.rollaxis(binimg.detach().cpu().numpy(), 0, 3) *255)

        if self.ret_boxes:
            pts = points
            return imgs, rots, trans, intrins, post_rots, post_trans, pts, binimg

        data = Sample(
                view=self.view,
                bev=binimg,
                **sample    
            )

        return data

def compile_data(dataroot, im_path, sessions, pitch=0, height=0, x=0, viz=False, rectify_data=False):
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
                    'H': 224, 'W': 480,
                    'cams': ['CAM_FRONT'],
                    'Ncams': 1,
                }
    data = DRIVESimDataset(dataroot,
                           im_path,
                           sessions,
                           data_aug_conf,
                           grid_conf,
                           ret_boxes=False,
                           pitch=pitch,
                           height=height,
                           x=x,
                           start=0,
                           stop=None,
                           viz=viz,
                           rectify_data=rectify_data)

    return data

if __name__ == "__main__":
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./DRIVESim_datasets/', 
                        help='path to top level data directory, i.e. DRIVESim_datasets')
    parser.add_argument("--session", type=str, default='5f8f235e-9304-11ed-8a70-6479f09080c1',
                        help='Session number (specified the folder containing all frames)')
    parser.add_argument("--dataset_idx", type=str, default='0', 
                        help='index of dataset, i.e. the number after 120fov_ (120fov_0, ... , 120fov_11)')
    parser.add_argument("--frames", type=str, default='rgb_half_jpeg-100-xavierisp', 
                        help='Either rgb_half_jpeg-100-xavierisp if loading non-rectified or rgb_jpeg_rectified if loading rectified')
    parser.add_argument("--vis", type=int, default=0,
                        help='whether or not to visualize data')
    parser.add_argument("--rectify", type=int, default=0,
                        help='whether or not to rectify data')
    args = parser.parse_args()

    im_path = os.path.join(args.dataroot, "frames", args.session, "camera_front_wide_120fov_" + args.dataset_idx, args.frames)
    pitch, height, x = DATASETS[args.dataset_idx]
    dataset = compile_data(args.dataroot, im_path, [args.session], pitch=pitch, height=height, x=x, viz=bool(args.vis), rectify_data=bool(args.rectify))

    # Only visualize first image ==> this can be changed to iterate the entire dataset if desired
    if args.vis:
        sample = dataset[0]
