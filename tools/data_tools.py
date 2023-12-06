"""
Copyright (C) 2023 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/NVlabs/viewpoint-robustness
Authors: Tzofi Klinghoffer, Jonah Philion, Wenzheng Chen, Or Litany, Zan Gojcic, Jungseock Joo, Ramesh Raskar, Sanja Fidler, Jose M. Alvarez
"""
import cv2
import torch
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Union
from tools.av_types import BBox2D, Cuboid3D, Cuboid3DGT, Center, Orientation, Dimension
from tools.transformations import euler_2_so3, transform_point_cloud, lat_lng_alt_2_ecef, axis_angle_trans_2_se3
from google import protobuf

import json
import base64
import logging
import os
import hashlib

import tqdm

import pyarrow.parquet as pq

from PIL import Image
from google.protobuf import text_format
from scipy.optimize import curve_fit
from numpy.polynomial.polynomial import Polynomial

ArrayLike = Union[torch.Tensor, np.ndarray]
MIN_ORIENTATION_ARROW_LENGTH = 3

def draw_3d_cuboids_on_bev_image(
    image: np.ndarray, cuboids: list, meter_to_pixel_scale: int = 4, thickness: int = 2
) -> np.ndarray:
    """Draw 3d cuboids on top-down image.
    Args:
        image (np.ndarray): an image of shape [W, H, 3] where objects will be drawn.
        cuboids: an list of 3d cuboids, where each cuboid is an numpy array
            of shape [10, 3] containing 10 3d vertices.
        meter_to_pixel_scale (int): Default value is 4.
        thickness (int): Default is 2.
    Outputs:
        image (np.ndarray): an image of shape [W, H, 3].
    """
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
        image = np.ascontiguousarray(image, dtype=np.uint8)
    img_width, img_height = image.shape[0:2]
    #for corners, source, class_id, visibility, _ in cuboids:
    for corners in cuboids:
        # Rotate 180 degree
        corners = -1 * corners
        # Convert meters to pixels
        corners = corners * meter_to_pixel_scale
        # Move center to top left
        corners[:, 1] += img_height / 2
        corners[:, 0] += img_width / 2
        # Swap x and y coordinates
        corners = np.int32(corners[:, [1, 0]].squeeze())
        #class_id = class_id.type(torch.uint8)
        boundary_color = [0.0, 1.0, 0.0]  # Green
        #COLOR_RGB_FP32[class_id]
        body_color = [0.0, 0.0, 1.0]  # Blue
        """
        if source == 1:
            body_color = [0.0, 0.0, 1.0]  # Blue
        elif source == 2:
            body_color = [0.0, 1.0, 0.0]  # Green
        else:
            body_color = cm.hot(100 + int((1 - visibility) * 200))[0:3]  # Red
        """
        cv2.fillPoly(image, [corners[0:4]], body_color)
        cv2.polylines(image, [corners[0:4]], True, boundary_color, thickness * 2)
        if corners.shape[0] > 8:
            start_point = corners[8, :]
            end_point = corners[9, :]
            cv2.arrowedLine(
                image, tuple(start_point), tuple(end_point), boundary_color, thickness
            )
    return image

def create_bev_image(
    max_x_viz_range_meters: int = 200,
    max_y_viz_range_meters: int = 100,
    meter_to_pixel_scale: int = 4,
    step_meters: int = 20,
    color: Optional[list] = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a bev image.
    Args:
        max_viz_range_meters (int): maximum visualization range. Default is 120 meters.
        meter_to_pixel_scale (int): Default is 4
        step_meters (int): steps to draw a spatial grid.
        color (tuple): self-explained.
        thickness (scalar): self-explained. Default is 2.
    Return:
        top_down_image (np.ndarray): an image of shape W x H x 3 where predictions are drawn.
    """
    # Create a back background image
    top_down_image = np.zeros(
        [
            2 * max_x_viz_range_meters * meter_to_pixel_scale,
            2 * max_y_viz_range_meters * meter_to_pixel_scale,
            3,
        ]
    )
    # Get bev center coordinates
    cx, cy = (
        max_y_viz_range_meters * meter_to_pixel_scale,
        max_x_viz_range_meters * meter_to_pixel_scale,
    )
    # Draw circular rings
    for d in range(step_meters, max_x_viz_range_meters, step_meters):
        radius = d * meter_to_pixel_scale
        ring_color = [1.0, 1.0, 1.0]
        text_color = [1.0, 1.0, 1.0]
        ring_thickness = 1
        cv2.circle(top_down_image, (cx, cy), radius, ring_color, ring_thickness)
        cv2.putText(
            top_down_image,
            str(d),
            (cx + 50, cy - radius),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2,
        )
    # Draw ego car
    scale = meter_to_pixel_scale
    ego_car_half_length = 2.0  # meters
    ego_car_half_width = 1.0  # meters
    ego_corners = np.array(
        [
            [cx - ego_car_half_width * scale, cy - ego_car_half_length * scale],
            [cx - ego_car_half_width * scale, cy + ego_car_half_length * scale],
            [cx + ego_car_half_width * scale, cy + ego_car_half_length * scale],
            [cx + ego_car_half_width * scale, cy - ego_car_half_length * scale],
        ],
        np.int32,
    )
    ego_corners = ego_corners.reshape((-1, 1, 2))
    ego_color = [1.0, 1.0, 1.0]
    cv2.fillPoly(top_down_image, [ego_corners], ego_color)
    ego_start = (cx, cy)
    ego_end = (cx, cy - int(3 * ego_car_half_length * meter_to_pixel_scale))
    cv2.arrowedLine(top_down_image, ego_start, ego_end, ego_color, thickness)
    return top_down_image

def numericallyStable2Norm2D(x, y):
    absX = abs(x)
    absY = abs(y)
    minimum = min(absX, absY)
    maximum = max(absX, absY)

    if maximum <= np.float32(0.0):
        return np.float32(0.0)

    oneOverMaximum = np.float32(1.0) / maximum
    minMaxRatio = np.float32(minimum) * oneOverMaximum
    return maximum * np.sqrt(np.float32(1.0) + minMaxRatio * minMaxRatio)

def project_camera_rays_2_img(points, cam_metadata):
    ''' Projects the points in the camera coordinate system to the image plane

    Args:
        points (np.array): point coordinates in the camera coordinate system [n,3]
        intrinsic (np.array): camera intrinsic parameters (size depends on the camera model)
        img_width (float): image width in pixels
        img_height (float): image hight in pixels
        camera_model (string): camera model used for projection. Must be one of ['pinhole', 'f_theta']
    Out:
        points_img (np.array): pixel coordinates of the image projections [m,2]
        valid (np.array): array of boolean flags. True for point that project to the image plane
    '''

    intrinsic = cam_metadata['intrinsic']
    camera_model = cam_metadata['camera_model']
    img_width = cam_metadata['img_width']
    img_height = cam_metadata['img_height']

    if camera_model == "pinhole":

        # Camera coordinates system is FLU and image is RDF
        normalized_points = -points[:,1:3] / points[:,0:1]
        f_u, f_v, c_u, c_v, k1, k2, k3, k4, k5 = intrinsic
        u_n = normalized_points[:,0]
        v_n = normalized_points[:,1]

        r2 = np.square(u_n) + np.square(v_n)
        r4 = r2 * r2
        r6 = r4 * r2

        r_d = 1.0 + k1 * r2 + k2 * r4 + k5 * r6

        # If the radial distortion is too large, the computed coordinates will be unreasonable
        kMinRadialDistortion = 0.8
        kMaxRadialDistortion = 1.2

        invalid_idx = np.where(np.logical_or(np.less_equal(r_d,kMinRadialDistortion),np.greater_equal(r_d,kMaxRadialDistortion)))[0]

        u_nd = u_n * r_d + 2.0 * k3 * u_n * v_n + k4 * (r2 + 2.0 * u_n * u_n)
        v_nd = v_n * r_d + k3 * (r2 + 2.0 * v_n * v_n) + 2.0 * k4 * u_n * v_n

        u_d = u_nd * f_u + c_u
        v_d = v_nd * f_v + c_v

        valid_flag = np.ones_like(u_d)
        valid_flag[points[:,0] <0] = 0

        # Replace the invalid ones
        r2_sqrt_rcp = 1.0 / np.sqrt(r2)
        clipping_radius = np.sqrt(img_width**2 + img_height**2)
        u_d[invalid_idx] = u_n[invalid_idx] * r2_sqrt_rcp[invalid_idx] * clipping_radius + c_u
        v_d[invalid_idx] = v_n[invalid_idx] * r2_sqrt_rcp[invalid_idx] * clipping_radius + c_v
        valid_flag[invalid_idx] = 0

        # Change the flags of the pixels that project outside of an image
        valid_flag[u_d < 0 ] = 0
        valid_flag[v_d < 0 ] = 0
        valid_flag[u_d > img_width] = 0
        valid_flag[v_d > img_height] = 0


        return np.concatenate((u_d[:,None], v_d[:,None]),axis=1),  np.where(valid_flag == 1)[0]

    elif camera_model == "f_theta":

        # Initialize the forward polynomial
        fw_poly = Polynomial(intrinsic[9:14])

        xy_norm = np.zeros((points.shape[0], 1))

        for i, point in enumerate(points):
            xy_norm[i] = numericallyStable2Norm2D(point[0], point[1])

        cos_alpha = points[:, 2:] / np.linalg.norm(points, axis=1, keepdims=True)
        alpha = np.arccos(np.clip(cos_alpha, -1 + 1e-7, 1 - 1e-7))
        delta = np.zeros_like(cos_alpha)
        valid = alpha <= intrinsic[16]

        delta[valid] = fw_poly(alpha[valid])

        # For outside the model (which need to do linear extrapolation)
        delta[~valid] = (intrinsic[14] + (alpha[~valid] - intrinsic[16]) * intrinsic[15])

        # Determine the bad points with a norm of zero, and avoid division by zero
        bad_norm = xy_norm <= 0
        xy_norm[bad_norm] = 1
        delta[bad_norm] = 0

        # compute pixel relative to center
        scale = delta / xy_norm
        pixel = scale * points

        # Handle the edge cases (ray along image plane normal)
        edge_case_cond = (xy_norm <= 0.0).squeeze()
        pixel[edge_case_cond, :] = points[edge_case_cond, :]
        points_img = pixel
        points_img[:, :2] += intrinsic[0:2]

        # Mark the points that do not fall on the camera plane as invalid
        x_ok = np.logical_and(0 <= points_img[:, 0], points_img[:, 0] < img_width)
        y_ok = np.logical_and(0 <= points_img[:, 1], points_img[:, 1] < img_height)
        z_ok = points_img[:,2] > 0.0
        valid = np.logical_and(np.logical_and(x_ok, y_ok), z_ok)

        return points_img, np.where(valid==True)[0]


def backwards_polynomial(pixel_norms, intrinsic):
    ret = 0
    for k, coeff in enumerate(intrinsic):

        ret += coeff * pixel_norms**k

    return ret

def pixel_2_camera_ray(pixel_coords, intrinsic, camera_model):
    ''' Convert the pixel coordinates to a 3D ray in the camera coordinate system.

    Args:
        pixel_coords (np.array): pixel coordinates of the selected points [n,2]
        intrinsic (np.array): camera intrinsic parameters (size depends on the camera model)
        camera_model (string): camera model used for projection. Must be one of ['pinhole', 'f_theta']

    Out:
        camera_rays (np.array): rays in the camera coordinate system [n,3]
    '''

    camera_rays = np.ones((pixel_coords.shape[0],3))

    if camera_model == 'pinhole':
        camera_rays[:,0] = (pixel_coords[:,0] + 0.5 - intrinsic[2]) / intrinsic[0]
        camera_rays[:,1] = (pixel_coords[:,1] + 0.5 - intrinsic[5]) / intrinsic[4]

    elif camera_model == "f_theta":
        pixel_offsets = np.ones((pixel_coords.shape[0],2))
        pixel_offsets[:,0] = pixel_coords[:,0] - intrinsic[0]
        pixel_offsets[:,1] = pixel_coords[:,1] - intrinsic[1]

        pixel_norms = np.linalg.norm(pixel_offsets, axis=1, keepdims=True)

        alphas = backwards_polynomial(pixel_norms, intrinsic[4:9])
        camera_rays[:,0:1] = (np.sin(alphas) * pixel_offsets[:,0:1]) / pixel_norms
        camera_rays[:,1:2] = (np.sin(alphas) * pixel_offsets[:,1:2]) / pixel_norms
        camera_rays[:,2:3] = np.cos(alphas)

        # special case: ray is perpendicular to image plane normal
        valid = (pixel_norms > np.finfo(np.float32).eps).squeeze()
        camera_rays[~valid, :] = (0, 0, 1)  # This is what DW sets these rays to

    return camera_rays

def compute_fw_polynomial(intrinsic):

    img_width = intrinsic[2]
    img_height = intrinsic[3]
    cxcy = np.array(intrinsic[0:2])

    max_value = 0.0
    value =  np.linalg.norm(np.asarray([0.0, 0.0], dtype=cxcy.dtype) - cxcy)
    max_value = max(max_value, value)
    value = np.linalg.norm(np.asarray([0.0, img_height], dtype=cxcy.dtype) - cxcy)
    max_value = max(max_value, value)
    value = np.linalg.norm(np.asarray([img_width, 0.0], dtype=cxcy.dtype) - cxcy)
    max_value = max(max_value, value)
    value = np.linalg.norm(np.asarray([img_width, img_height], dtype=cxcy.dtype) - cxcy)
    max_value = max(max_value, value)

    SAMPLE_COUNT = 500
    samples_x = []
    samples_b = []
    step = max_value / SAMPLE_COUNT
    x = step

    for _ in range(0, SAMPLE_COUNT):
        p = np.asarray([cxcy[0] + x, cxcy[1]], dtype=np.float64).reshape(-1,2)
        ray = pixel_2_camera_ray(p, intrinsic, 'f_theta')
        xy_norm = np.linalg.norm(ray[0, :2])
        theta = np.arctan2(float(xy_norm), float(ray[0, 2]))
        samples_x.append(theta)
        samples_b.append(float(x))
        x += step

    x = np.asarray(samples_x, dtype=np.float64)
    y = np.asarray(samples_b, dtype=np.float64)
    # Fit a 4th degree polynomial. The polynomial function is as follows:

    def f(x, b, x1, x2, x3, x4):
        """4th degree polynomial."""
        return b + x1 * x + x2 * (x ** 2) + x3 * (x ** 3) + x4 * (x ** 4)

    # The constant in the polynomial should be zero, so add the `bounds` condition.
    coeffs, _ = curve_fit(
        f,
        x,
        y,
        bounds=(
            [0, -np.inf, -np.inf, -np.inf, -np.inf],
            [np.finfo(np.float64).eps, np.inf, np.inf, np.inf, np.inf],
        ),
    )

    return np.array([np.float32(val) if i > 0 else 0 for i, val in enumerate(coeffs)], dtype=np.float32)


def compute_ftheta_fov(intrinsic):
    """Computes the FOV of this camera model."""
    max_x = intrinsic[2] - 1
    max_y = intrinsic[3] - 1

    point_left = np.asarray([0.0, intrinsic[1]]).reshape(-1,2)
    point_right = np.asarray([max_x, intrinsic[1]]).reshape(-1,2)
    point_top = np.asarray([intrinsic[0], 0.0]).reshape(-1,2)
    point_bottom = np.asarray([intrinsic[0], max_y]).reshape(-1,2)

    fov_left = _get_pixel_fov(point_left, intrinsic)
    fov_right = _get_pixel_fov(point_right, intrinsic)
    fov_top = _get_pixel_fov(point_top, intrinsic)
    fov_bottom = _get_pixel_fov(point_bottom, intrinsic)

    v_fov = fov_top + fov_bottom
    hz_fov = fov_left + fov_right
    max_angle = _compute_max_angle(intrinsic)

    return v_fov, hz_fov, max_angle


def _get_pixel_fov(pt, intrinsic):
    """Gets the FOV for a given point. Used internally for FOV computation of the F-theta camera.

    Args:
        pt (np.ndarray): 2D pixel.

    Returns:
        fov (float): the FOV of the pixel.
    """
    ray = pixel_2_camera_ray(pt, intrinsic, 'f_theta')
    fov = np.arctan2(np.linalg.norm(ray[:, :2], axis=1), ray[:, 2])
    return fov


def _compute_max_angle(intrinsic):

    p = np.asarray(
        [[0, 0], [intrinsic[2] - 1, 0], [0, intrinsic[3] - 1], [intrinsic[2] - 1, intrinsic[3] - 1]], dtype=np.float32
    )

    return max(
        max(_get_pixel_fov(p[0:1, ...], intrinsic), _get_pixel_fov(p[1:2, ...], intrinsic)),
        max(_get_pixel_fov(p[2:3, ...], intrinsic), _get_pixel_fov(p[3:4, ...], intrinsic)),
    )

def get_sensor_to_sensor_flu(sensor):
    """Compute a rotation transformation matrix that rotates sensor to Front-Left-Up format.

    Args:
        sensor (str): sensor name.

    Returns:
        np.ndarray: the resulting rotation matrix.
    """
    if "cam" in sensor:
        rot = [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    else:
        rot = np.eye(4, dtype=np.float32)

    return np.asarray(rot, dtype=np.float32)

def parse_rig_sensors_from_dict(rig):
    """Parses the provided rig dictionary into a dictionary indexed by sensor name.

    Args:
        rig (Dict): Complete rig file as a dictionary.

    Returns:
        (Dict): Dictionary of sensor rigs indexed by sensor name.
    """
    # Parse the properties from the rig file
    sensors = rig["rig"]["sensors"]

    sensors_dict = {sensor["name"]: sensor for sensor in sensors}
    return sensors_dict


def parse_rig_sensors_from_file(rig_fp):
    """Parses the provided rig file into a dictionary indexed by sensor name.

    Args:
        rig_fp (str): Filepath to rig file.

    Returns:
        (Dict): Dictionary of sensor rigs indexed by sensor name.
    """
    # Read the json file
    with open(rig_fp, "r") as fp:
        rig = json.load(fp)

    return parse_rig_sensors_from_dict(rig)


def sensor_to_rig(sensor):

    sensor_name = sensor["name"]
    sensor_to_FLU = get_sensor_to_sensor_flu(sensor_name)

    nominal_T = sensor["nominalSensor2Rig_FLU"]["t"]
    nominal_R = sensor["nominalSensor2Rig_FLU"]["roll-pitch-yaw"]

    correction_T = np.zeros(3, dtype=np.float32)
    correction_R = np.zeros(3, dtype=np.float32)

    if ("correction_rig_T" in sensor.keys()):
        correction_T = sensor["correction_rig_T"]

    if ("correction_sensor_R_FLU" in sensor.keys()):
        assert "roll-pitch-yaw" in sensor["correction_sensor_R_FLU"].keys(), str(sensor["correction_sensor_R_FLU"])
        correction_R = sensor["correction_sensor_R_FLU"]["roll-pitch-yaw"]

    nominal_R = euler_2_so3(nominal_R)
    correction_R = euler_2_so3(correction_R)

    #from worldsheet.render_utils import rotationMatrixToEulerAngles
    #from transformations import so3_2_axis_angle
    #print(np.degrees(rotationMatrixToEulerAngles(nominal_R)))
    #print(np.degrees(rotationMatrixToEulerAngles(correction_R)))
    R = nominal_R @ correction_R
    #print(np.degrees(rotationMatrixToEulerAngles(R)))
    #exit()
    T =  np.array(nominal_T, dtype=np.float32) + np.array(correction_T, dtype=np.float32)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R
    transform[:3, 3] = T

    sensor_to_rig = transform @ sensor_to_FLU

    return sensor_to_rig, R


def camera_intrinsic_parameters(sensor: dict,
                                logger: Optional[logging.Logger] = None
                                ) -> np.ndarray:
    """  Parses the provided rig-style camera sensor dictionary into FTheta camera intrinsic parameters.

    Note: currenlty only 5th-order 'pixeldistance-to-angle' ("bw-poly") FTheta are supported, possibly
          available 6th-order term will be dropped with a warning

    Args:
        sensor: the dictionary of the sensor parameters read from the rig file
        logger: if provided, the logger to issue warnings in (e.g., on not supported coeffiecients)
    Returns:
        intrinsic: array of FTheta intrinsics [cx, cy, width, height, [bwpoly]]
    """

    assert sensor['properties'][
        'Model'] == 'ftheta', "unsupported camera model (only supporting FTheta)"

    cx = float(sensor['properties']['cx'])
    cy = float(sensor['properties']['cy'])
    width = float(sensor['properties']['width'])
    height = float(sensor['properties']['height'])

    if 'bw-poly' in sensor['properties']:
        # Legacy 5-th order backwards-polynomial
        bwpoly = [
            np.float32(val) for val in sensor['properties']['bw-poly'].split()
        ]
        assert len(
            bwpoly
        ) == 5, "expecting fifth-order coefficients for 'bw-poly / 'pixeldistance-to-angle' polynomial"
    elif 'polynomial' in sensor['properties']:
        # Two-way forward / backward polynomial encoding
        assert sensor['properties']['polynomial-type'] == 'pixeldistance-to-angle', \
            f"currently only supporting 'pixeldistance-to-angle' polynomial type, received '{sensor['properties']['polynomial-type']}'"

        bwpoly = [
            np.float32(val)
            for val in sensor['properties']['polynomial'].split()
        ]

        if len(bwpoly) > 5:
            # WAR: 6th-order polynomials are currently not supported in the software-stack, drop highest order coeffient for now
            # TODO: extend internal camera model and NGP with support for 6th-order polynomials
            if logger:
                logger.warn(
                    f"> encountered higher-order distortion polynomial for camera '{sensor['name']}', restricting to 5th-order, dropping coefficients '{bwpoly[5:]}' - parsed model might be inaccurate"
                )

            bwpoly = bwpoly[:5]

        # Affine term is currently not supported, issue a warning if it differs from identity
        # TODO: properly incorporate c,d,e coefficients of affine term [c, d; e, 1] into software stack (internal camera models + NGP)
        A = np.matrix([[np.float32(sensor['properties'].get('c', 1.0)), np.float32(sensor['properties'].get('d', 0.0))], \
                        [np.float32(sensor['properties'].get('e', 0.0)), np.float32(1.0)]])

        if (A != np.identity(2, dtype=np.float32)).any():
            if logger:
                logger.warn(
                    f"> *not* considering non-identity affine term '{A}' for '{sensor['name']}' - parsed model might be inaccurate"
                )

    else:
        raise ValueError("unsupported distortion polynomial type")

    intrinsic = [cx, cy, width, height] + bwpoly

    return np.array(intrinsic, dtype=np.float32)

def degree_to_radian(angles: ArrayLike):
    """Convert angles in degrees to radians.
    Args:
        angles (torch.Tensor | np.ndarray): An array in degrees.
    Returns:
        angles (torch.Tensor | np.ndarray): An array in radians. Range will be [0, 2 * pi).
    """
    if isinstance(angles, np.ndarray):
        opset = np
    elif isinstance(angles, torch.Tensor):
        opset = torch
    else:
        raise TypeError("Unsupported data type.")
    angles = opset.remainder(angles, 360)
    angles = opset.divide(angles * np.pi, 180)
    return angles

def euler_to_rotation_matrix(euler_angles: ArrayLike) -> np.ndarray:
    """Convert euler angles to a rotation matrix.
    Args:
        euler_angles: An array of euler angles using the Tait-Bryan convention
            of yaw, pitch, roll (the ZYX sequence of intrinsic rotations), as
            described in:
            https://en.wikipedia.org/wiki/Euler_angles#Conventions.
            Angles are assumed to be in radians.
    Returns:
        rotation_matrix: A 3x3 rotation matrix as a numpy array.
    """
    sin_yaw, sin_pitch, sin_roll = np.sin(euler_angles)
    cos_yaw, cos_pitch, cos_roll = np.cos(euler_angles)
    # Calculations according to the Tait-Bryan column and ZYX row of the
    # following table:
    # https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    rotation_matrix = np.array(
        [
            [
                cos_yaw * cos_pitch,
                cos_yaw * sin_pitch * sin_roll - cos_roll * sin_yaw,
                sin_yaw * sin_roll + cos_yaw * cos_roll * sin_pitch,
            ],
            [
                cos_pitch * sin_yaw,
                cos_yaw * cos_roll + sin_yaw * sin_pitch * sin_roll,
                cos_roll * sin_yaw * sin_pitch - cos_yaw * sin_roll,
            ],
            [-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll],
        ]
    )
    return rotation_matrix

def mrp_to_quat(mrp: ArrayLike) -> ArrayLike:
    """MRP to Quat.
    Convert modified Rodrigues parameters (MRP)
    representation to quaternion rotation.
    Args:
        mrp (torch.Tensor | np.ndarray): MRP based rotation
    Return:
        quat (torch.Tensor | np.ndarray): quaternion based rotation
    """
    if isinstance(mrp, np.ndarray):
        opset = np
    elif isinstance(mrp, torch.Tensor):
        opset = torch
    else:
        raise TypeError("Unsupported data type for MRP.")
    if len(mrp.shape) == 1:
        # mrp is a 1d array
        assert mrp.shape[0] == 3
        quat = opset.zeros(4, dtype=mrp.dtype)
        mrp_squared_plus = 1 + mrp[0] ** 2 + mrp[1] ** 2 + mrp[2] ** 2
        quat[0] = 2 * mrp[0] / mrp_squared_plus
        quat[1] = 2 * mrp[1] / mrp_squared_plus
        quat[2] = 2 * mrp[2] / mrp_squared_plus
        quat[3] = (2 - mrp_squared_plus) / mrp_squared_plus
    elif len(mrp.shape) == 2:
        assert mrp.shape[1] == 3
        quat = opset.zeros((mrp.shape[0], 4), dtype=mrp.dtype)
        mrp_squared_plus = 1 + mrp[:, 0:1] ** 2 + mrp[:, 1:2] ** 2 + mrp[:, 2:3] ** 2
        quat[:, 0:1] = 2 * mrp[:, 0:1] / mrp_squared_plus
        quat[:, 1:2] = 2 * mrp[:, 1:2] / mrp_squared_plus
        quat[:, 2:3] = 2 * mrp[:, 2:3] / mrp_squared_plus
        quat[:, 3:4] = (2 - mrp_squared_plus) / mrp_squared_plus
    return quat

def quat_to_rotation(quat: ArrayLike) -> ArrayLike:
    """Quat to Rotation matrix conversion.
    Convert quaternion rotation representation to Rotation matrix.
    Args:
        quat (torch.Tensor | np.ndarray): quaternion based rotation
    Return:
        rot_matrix (torch.Tensor | np.ndarray): 3x3 rotation matrix
    """
    # TOTO(bala) Looks like this quat to rotation implementation is inconsistent with
    # scipy.spatial.transform.Rotation.from_quat
    # Jira AMP-313
    if isinstance(quat, np.ndarray):
        opset = np
    elif isinstance(quat, torch.Tensor):
        opset = torch
    else:
        raise TypeError("Unsupported data type.")
    # Order is changed to match the convention of scipy.spatial.transform.Rotation.from_quat
    q0 = quat[3]
    q1 = quat[0]
    q2 = quat[1]
    q3 = quat[2]
    rot_matrix = opset.zeros((3, 3), dtype=quat.dtype)
    # First row of the rotation matrix
    rot_matrix[0][0] = 1 - 2 * (q2 * q2 + q3 * q3)
    rot_matrix[0][1] = 2 * (q1 * q2 - q0 * q3)
    rot_matrix[0][2] = 2 * (q1 * q3 + q0 * q2)
    # Second row of the rotation matrix
    rot_matrix[1][0] = 2 * (q1 * q2 + q0 * q3)
    rot_matrix[1][1] = 1 - 2 * (q1 * q1 + q3 * q3)
    rot_matrix[1][2] = 2 * (q2 * q3 - q0 * q1)
    # Third row of the rotation matrix
    rot_matrix[2][0] = 2 * (q1 * q3 - q0 * q2)
    rot_matrix[2][1] = 2 * (q2 * q3 + q0 * q1)
    rot_matrix[2][2] = 1 - 2 * (q1 * q1 + q2 * q2)
    # 3x3 rotation matrix
    return rot_matrix

def compute_cuboid_corners(
    cuboid: ArrayLike, use_polar: bool = False, use_mrp: bool = True
) -> ArrayLike:
    """Compute 8 cuboid vertices.
    Args:
        cuboid (torch.Tensor | np.ndarray): It can be an array of 9 elements
            (x/radius, y/angle, z, w, l, h, mrp_0/x_angle, mrp_1/y_angle, mrp_2/z_angle)
            representing a 3d cuboid. (mrp_0, mrp_1, mrp_2) triplet is the MRP representation from
            https://link.springer.com/content/pdf/10.1007/s10851-017-0765-x.pdf
            Or it is also can be an array of 15 elements
            (x/radius, y/angle, z, w, l, h, 9 elements in rotation matrix).
        use_polar (bool): flag whether center_x/center_y is represented polar coordinates.
            Default `False` means using Cartesian coordinates.
        use_mrp (bool): flag whether the rotation is represented as MRP (see above for details)
            or as (yaw,pitch,roll). Default `True` means MRP representation is expected.
    Returns:
        corners ((torch.Tensor | np.ndarray)): an array of 3d vertices of shape [8, 3]
    """
    if isinstance(cuboid, np.ndarray):
        opset = np
    elif isinstance(cuboid, torch.Tensor):
        opset = torch
    else:
        raise TypeError("Unsupported data type.")
    if cuboid.shape[0] == 15:
        rot = cuboid[6::].reshape(3, 3)
    else:
        rot = cuboid[6:9]
        if use_mrp:
            rot = quat_to_rotation(mrp_to_quat(rot))
        else:
            rot = euler_to_rotation_matrix(rot)
    if use_polar:
        u = cuboid[0]
        v = cuboid[1]
        center_x = u * opset.cos(v)
        center_y = u * opset.sin(v)
    else:
        center_x = cuboid[0]
        center_y = cuboid[1]
    center_z = cuboid[2]
    dim_0 = cuboid[3] / 2
    dim_1 = cuboid[4] / 2
    dim_2 = cuboid[5] / 2
    t = opset.zeros((3, 1), dtype=cuboid.dtype)
    t[0][0] = center_x
    t[1][0] = center_y
    t[2][0] = center_z
    corners = opset.zeros((3, 10), dtype=cuboid.dtype)
    corners[0, 0] = center_x + dim_0  # Front, Left, Top
    corners[1, 0] = center_y + dim_1
    corners[2, 0] = center_z + dim_2
    corners[0, 1] = center_x + dim_0  # Front, right, top
    corners[1, 1] = center_y - dim_1
    corners[2, 1] = center_z + dim_2
    corners[0, 2] = center_x - dim_0  # Back, right, top
    corners[1, 2] = center_y - dim_1
    corners[2, 2] = center_z + dim_2
    corners[0, 3] = center_x - dim_0  # Back, left, top
    corners[1, 3] = center_y + dim_1
    corners[2, 3] = center_z + dim_2
    corners[0, 4] = center_x + dim_0  # Front, left, bot
    corners[1, 4] = center_y + dim_1
    corners[2, 4] = center_z - dim_2
    corners[0, 5] = center_x + dim_0  # Front, right, bot
    corners[1, 5] = center_y - dim_1
    corners[2, 5] = center_z - dim_2
    corners[0, 6] = center_x - dim_0  # Back, right, bot
    corners[1, 6] = center_y - dim_1
    corners[2, 6] = center_z - dim_2
    corners[0, 7] = center_x - dim_0  # Back, leftt, bot
    corners[1, 7] = center_y + dim_1
    corners[2, 7] = center_z - dim_2
    corners[0, 8] = center_x
    corners[1, 8] = center_y
    corners[2, 8] = center_z
    corners[0, 9] = center_x + max(MIN_ORIENTATION_ARROW_LENGTH, 2 * dim_0)
    corners[1, 9] = center_y
    corners[2, 9] = center_z
    corners = opset.matmul(rot, (corners - t)) + t
    corners = corners.T
    return corners

def shape2d_flatten_vertices(shape2d):
    """Retrieves and flattens the vertices of a Shape2D object into a single list of pairs.
    Args:
        shape2d(protobuf): Shape2D protobuf with  Repeated field of vertices.
    Returns:
        list of list of floats.
    """
    vertices = getattr(shape2d, shape2d.WhichOneof("shape2d")).vertices
    return [[vertex.x, vertex.y] for vertex in vertices]

def shape2d_flatten_cuboid2d_vertices(cuboid2d):
    """Flattens the vertices into a single list pairs as expected by the rest of the code.
    Args:
        cuboid2d(protobuf): cuboid2d protobuf with repeated field of vertices.
    Returns:
        list of list of floats.
    """
    return [[vertex.x, vertex.y] for vertex in cuboid2d]

def attribute_as_dict(attributes: protobuf.pyext._message.RepeatedCompositeContainer):
    """Converts interchange attributes in a label to a dict mapping attribute name to value.
    Handles extraction of the attribute based on the type of the attribute and extracts the list
    out in the case where it's an enum_list.
    Args:
        attributes: (proto) label with attributes to be extracted.
    Returns:
        Dict: parsed dictionary mapping attribute.name to attribute.value
    """
    attribute_dict = {}
    for attribute in attributes:
        which_one = attribute.WhichOneof("attr")
        value = getattr(attribute, which_one)
        if which_one == "enums_list":
            value = value.enums_list
        attribute_dict[attribute.name] = value
    return attribute_dict

def parse(features, frames) -> Cuboid3DGT:
        """Parse method for camera features.
        Args:
            scene (Dict): A dictionary mapping a sensor name to a tuple consisting of the list
                of Features belonging to that frame and a Frame object containing metadata about the
                sensor frame.
        Raises:
            AssertionError: If scene doesn't contain a sensor defined in the feature_sensors list.
            AssertionError: If label is not in interchange format.
        Returns:
            result (Cuboid3DGT): A collection of the cuboid3d labels accumulated over all feature
                rows corresponding to this scene.
        """
        """
        if row.label_data_type == LabelDataType.from_string("SHAPE2D:BOX2D"):
                shape2d = row.data.shape2d
                for attr in shape2d.attributes:
                    if attr.name == "cuboid3d_rig":
                        cuboid3d = Cuboid3D.from_ilf(attr.cuboid3d)
        """

        result = Cuboid3DGT(*[[] for _ in Cuboid3DGT._fields])
        default_box2ds = {}
        for row in features:
            if row.label_data_type == LabelDataType.from_string("SHAPE2D:BOX2D"):
                shape3d = row.data.shape2d
                attributes = attribute_as_dict(shape3d.attributes)
                label_name = attributes.get("label_name", None)
                occlusion_ratio = attributes.get("occlusion_ratio", None)
                #print(label_name)
                if "vehicle" not in label_name and "truck" not in label_name and "automobile" not in label_name:
                    continue
                #print(attributes.get("cuboid3d_rig"))
                #print(type(attributes.get("cuboid3d_rig")))
                #exit()
                cuboid3d = Cuboid3D.from_ilf(attributes.get("cuboid3d_rig", None))
                result.cuboid3ds.append(cuboid3d)
                result.visibility.append(occlusion_ratio)
        return result

def parse_drivesim(data) -> Cuboid3DGT:
    cuboid3d = None
    ratio = None
    for attr in data["shape2d"]["attributes"]:
        if attr["name"] == "occlusion_ratio":
            ratio = attr["numerical"]
        elif attr["name"] == "cuboid3d_rig":
            cuboid3d = attr["cuboid3d"]
            center = Center(cuboid3d['center']['x'], cuboid3d['center']['y'], cuboid3d['center']['z'])
            orientation = Orientation(
                cuboid3d['orientation']['x'], cuboid3d['orientation']['y'], cuboid3d['orientation']['z']
            )
            dimension = Dimension(
                cuboid3d['dimensions']['x'], cuboid3d['dimensions']['y'], cuboid3d['dimensions']['z']
            )
            cuboid3d = Cuboid3D(center=center, orientation=orientation, dimension=dimension)
    return cuboid3d, ratio

    """
    cuboid3d = data["shape3d"]["cuboid3d"]
    center = Center(cuboid3d['center']['x'], cuboid3d['center']['y'], cuboid3d['center']['z'])
    orientation = Orientation(
        cuboid3d['orientation']['x'], cuboid3d['orientation']['y'], cuboid3d['orientation']['z']
    )
    dimension = Dimension(
        cuboid3d['dimensions']['x'], cuboid3d['dimensions']['y'], cuboid3d['dimensions']['z']
    )
    cuboid3d = Cuboid3D(center=center, orientation=orientation, dimension=dimension)
    visibility = None 
    for attr in data["shape3d"]["attributes"]:
        if attr["name"] == "camera_front_wide_120fov_visibility":
            visibility = attr["numerical"]
    return cuboid3d, visibility
    """

"""
def read_2d_boxes(shape2d, frames: Frame, denormalize=True) -> Optional[BBox2D]:
    if False: #self._use_human_label:
        vertices = shape2d_flatten_vertices(shape2d)
        x_min, x_max = vertices[0][0], vertices[1][0]
        y_min, y_max = vertices[0][1], vertices[1][1]
    else:
        attributes = attribute_as_dict(shape2d.attributes)
        cuboid2d = attributes.get("cuboid2d", None)
        if cuboid2d is None:
            return None
        cuboid_vertices = shape2d_flatten_cuboid2d_vertices(
            cuboid2d.vertices_object
        )
        if cuboid_vertices == []:
            return None
        x_min = min(cuboid_vertices, key=lambda x: x[0])[0]
        x_max = max(cuboid_vertices, key=lambda x: x[0])[0]
        y_min = min(cuboid_vertices, key=lambda x: x[1])[1]
        y_max = max(cuboid_vertices, key=lambda x: x[1])[1]
    if denormalize:
        x_min = x_min * frames.original_width
        x_max = x_max * frames.original_width
        y_min = y_min * frames.original_height
        y_max = y_max * frames.original_height
    box2d = BBox2D.from_ilf([[x_min, y_min], [x_max, y_max]])
    return box2d
"""
