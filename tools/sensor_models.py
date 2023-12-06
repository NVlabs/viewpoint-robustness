"""
Copyright (C) 2023 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/NVlabs/viewpoint-robustness
Authors: Tzofi Klinghoffer, Jonah Philion, Wenzheng Chen, Or Litany, Zan Gojcic, Jungseock Joo, Ramesh Raskar, Sanja Fidler, Jose M. Alvarez
"""

"""Camera model definitions."""
import cv2
import json
import logging
from typing import Any, Dict, Tuple, Union
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit
logger = logging.getLogger("dlav.autonet_v2.sdk.sensors.camera.camera_models")


class IdealPinholeCamera:
    """Reperesents an ideal pinhole camera with no distortions.
    You can either pass in the fov or you can pass in the actual
    focal point parameters. It is the users choice. If you pass
    in the fov, then the f_x, f_y parameters are computed for you.
    Otherwise, they are directly inserted into the intrinsic matrix.
    """
    def __init__(
        self,
        fov_x_deg: Union[float, int] = None,
        fov_y_deg: Union[float, int] = None,
        f_x: Union[float, int] = None,
        f_y: Union[float, int] = None,
        width: int = 3848,
        height: int = 2168,
    ):
        """The __init__ function.
        Args:
            fov_x_deg (Union[float, int]): the horizontal FOV in degrees.
            fov_y_deg (Union[float, int]): the vertical FOV in degrees.
            f_x (Union[float, int]): the f_x value of the intrinsic calibration
                matrix
            f_y (Union[float, int]): the f_y value of the intrinsic calibration
                matrix
            width (int): the width of the image. Defaults to 3848
            height (int): the height of the image. Defaults to 2168
        """
        if f_x and fov_x_deg or f_y and fov_y_deg:
            raise ValueError(
                "Either f_x,f_y or fov_x_deg, fov_y_deg can"
                "be passed in but not both. User must select which"
                "operational mode you intend to use. If you want to"
                "directly insert fx,fy into the intrinsic calibration"
                "matrix then do not pass in fov_x_deg or fov_y_deg"
                "and if you want to compute f_x, f_y from the FOV then"
                "do not pass in f_x, f_y"
            )
        self._width = width
        self._height = height
        self._cx = width / 2
        self._cy = height / 2
        # You can pass in the values directly.
        if f_x and f_y:
            self._f_x = f_x
            self._f_y = f_y
        else:
            self._focal_from_fov(fov_x_deg, fov_y_deg)
        # The intrinsics matrix
        self._k = np.asarray(
            [[self._f_x, 0, self._cx], [0, self._f_y, self._cy], [0, 0, 1]],
            dtype=np.float32,
        )
        # The inverse of the intrinsics matrix (for backprojection)
        self._k_inv = np.asarray(
            [
                [1.0 / self._f_x, 0, -self._cx / self._f_x],
                [0, 1.0 / self._f_y, -self._cy / self._f_y],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
    @property
    def width(self) -> int:
        """Returns the width of the sensor."""
        return self._width
    @property
    def height(self) -> int:
        """"Returns the height of the sensor."""
        return self._height
    def _focal_from_fov(
        self, fov_x_deg: Union[float, int], fov_y_deg: Union[float, int]
    ):
        """Compute the focal length from horizontal and vertical FOVs.
        Args:
            fov_x_deg (Union[float, int]): the horizontal FOV in degrees.
            fov_y_deg (Union[float, int]): the vertical FOV in degrees.
        """
        fov_x = np.radians(fov_x_deg)
        fov_y = np.radians(fov_y_deg)
        self._f_x = self._width / (2.0 * np.tan(fov_x * 0.5))
        self._f_y = self._height / (2.0 * np.tan(fov_y * 0.5))
    def ray2pixel(self, rays: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D rays to 2D pixel coordinates.
        Args:
            rays (np.ndarray): the rays as (N, 3) where N corresponds to
                the number of rays and 3 is the (x,y,z) coordinates for each
                ray.
        Returns:
            projected (np.ndarray): Shape (N,2) the projected pixel coordinates
                where N is the number of points and 2 corresponds to the (x,y) dimensions.
            valid (np.ndarray): of Shape (N,) the validity flag for each projected pixel.
                Valid is a boolean array that can be used for indexing rays
                that are within FOV.
        """
        if np.ndim(rays) == 1:
            rays = rays[np.newaxis, :]
        rays = rays.astype(np.float32)
        r = rays / rays[:, 2:]
        projected = np.matmul(self._k, r.T).T
        x_ok = np.logical_and(0 <= projected[:, 0], projected[:, 0] < self._width)
        y_ok = np.logical_and(0 <= projected[:, 1], projected[:, 1] < self._height)
        valid = np.logical_and(x_ok, y_ok)
        return projected[:, :2], valid
    def pixel2ray(self, pixels: np.ndarray) -> np.ndarray:
        """Backproject 2D pixels into 3D rays.
        Args:
            pixels (np.ndarray): the pixels to backproject. Size of (n_points, 2), where the first
                column contains the `x` values, and the second column contains the `y` values.
        Returns:
            rays (np.ndarray): the backprojected 3D rays.
        """
        if np.ndim(pixels) == 1:
            pixels = pixels[np.newaxis, :]
        pixels = pixels.astype(np.float32)
        # Add the third component of ones
        pixels = np.c_[pixels, np.ones((pixels.shape[0], 1), dtype=np.float32)]
        rays = np.matmul(self._k_inv, pixels.T).T
        # Normalize the rays
        norm = np.linalg.norm(rays, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return rays / norm
class FThetaCamera:
    """Defines an FTheta camera model."""
    @classmethod
    def from_rig(cls, rig_file: str, sensor_name: str):
        """Helper method to initialize a new object using a rig file and the sensor's name.
        Args:
            rig_file (str): the rig file path.
            sensor_name (str): the name of the sensor.
        Returns:
            FThetaCamera: the newly created object.
        """
        with open(rig_file, "r") as fp:
            rig = json.load(fp)
        # Parse the properties from the rig file
        sensors = rig["rig"]["sensors"]
        sensor = None
        sensor_found = False
        for sensor in sensors:
            if sensor["name"] == sensor_name:
                sensor_found = True
                break
        if not sensor_found:
            raise ValueError(f"The camera '{sensor_name}' was not found in the rig!")
        return cls.from_dict(sensor)
    @classmethod
    def from_dict(cls, rig_dict: Dict[str, Any]):
        """Helper method to initialize a new object using a dictionary of the rig.
        Args:
            rig_dict (dict): the sensor dictionary to initialize with.
        Returns:
            FThetaCamera: the newly created object.
        """
        cx, cy, width, height, bw_poly = FThetaCamera.get_ftheta_parameters_from_json(
            rig_dict
        )
        return cls(cx, cy, width, height, bw_poly)
    @classmethod
    def from_intrinsics_array(cls, intrinsics: np.ndarray):
        """Helper method to initialize a new object using an array of intrinsics.
        Args:
            intrinsics (np.ndarray): the intrinsics array. The ordering is expected to be
                "cx, cy, width, height, bw_poly". This is the same ordering as the `intrinsics`
                property of this class.
        Returns:
            FThetaCaamera: the newly created object.
        """
        return cls(
            cx=intrinsics[0],
            cy=intrinsics[1],
            width=intrinsics[2],
            height=intrinsics[3],
            bw_poly=intrinsics[4:],
        )
    def __init__(
        self, cx: float, cy: float, width: int, height: int, bw_poly: np.ndarray
    ):
        """The __init__ method.
        Args:
            cx (float): optical center x.
            cy (float): optical center y.
            width (int): the width of the image.
            height (int): the height of the image.
            bw_poly (np.ndarray): the backward polynomial of the FTheta model.
        """
        self._center = np.asarray([cx, cy], dtype=np.float32)
        self._width = int(width)
        self._height = int(height)
        assert isinstance(bw_poly, np.ndarray) or isinstance(bw_poly, list)
        self.bw_poly = Polynomial(bw_poly)
        self._fw_poly = self._compute_fw_poly()
        # Other properties that need to be computed
        self._horizontal_fov = None
        self._vertical_fov = None
        self._max_angle = None
        self._max_ray_angle = None
        # Populate the array of intrinsics
        self._intrinsics = np.append([cx, cy, width, height], bw_poly).astype(
            np.float32
        )
        self._update_calibrated_camera()
    @staticmethod
    def get_ftheta_parameters_from_json(rig_dict: Dict[str, Any]) -> Tuple[Any]:
        """Helper method for obtaining FTheta camera model parameters from a rig dict.
        Args:
            rig_dict (Dict[str, Any]): the rig dictionary to parse.
        Raises:
            ValueError: if the provided rig is not supported.
            AssertionError: if the provided model is supported, but cannot be parsed properly.
        Returns:
            Tuple[Any]: the values `cx`, `cy`, `width`, `height` and `bw_poly` that were parsed.
        """
        props = rig_dict["properties"]
        if props["Model"] != "ftheta":
            raise ValueError("The given camera is not an FTheta camera")
        cx = float(props["cx"])
        cy = float(props["cy"])
        width = int(props["width"])
        height = int(props["height"])
        if "bw-poly" in props:  # Is this a regular rig?
            poly = props["bw-poly"]
        elif "polynomial" in props:  # Is this a VT rig?
            # VT rigs have a slightly different format, so need to handle these
            # specifically. Refer to the following thread for more details:
            # https://nvidia.slack.com/archives/C017LLEG763/p1633304770105300
            poly_type = props["polynomial-type"]
            assert poly_type == "pixeldistance-to-angle", (
                "Encountered an unsupported VT rig. Only `pixeldistance-to-angle` "
                f"polynomials are supported (got {poly_type}). Rig:\n{rig_dict}"
            )
            linear_c = float(props["linear-c"]) if "linear-c" in props else None
            linear_d = float(props["linear-d"]) if "linear-d" in props else None
            linear_e = float(props["linear-e"]) if "linear-e" in props else None
            # If we had all the terms present, sanity check to make sure they are [1, 0, 0]
            if linear_c is not None and linear_d is not None and linear_e is not None:
                assert (
                    linear_c == 1.0
                ), f"Expected `linear-c` term to be 1.0 (got {linear_c}. Rig:\n{rig_dict})"
                assert (
                    linear_d == 0.0
                ), f"Expected `linear-d` term to be 0.0 (got {linear_d}. Rig:\n{rig_dict})"
                assert (
                    linear_e == 0.0
                ), f"Expected `linear-e` term to be 0.0 (got {linear_e}. Rig:\n{rig_dict})"
            # If we're here, then it means we can parse the rig successfully.
            poly = props["polynomial"]
        else:
            raise ValueError(
                f"Unable to parse the rig. Only FTheta rigs are supported! Rig:\n{rig_dict}"
            )
        bw_poly = [np.float32(val) for val in poly.split()]
        return cx, cy, width, height, bw_poly
    @property
    def fov(self) -> tuple:
        """Returns a tuple of horizontal and vertical fov of the sensor."""
        if self._vertical_fov is None or self._horizontal_fov is None:
            self._compute_fov()
        return self._horizontal_fov, self._vertical_fov
    @property
    def width(self) -> int:
        """Returns the width of the sensor."""
        return self._width
    @property
    def height(self) -> int:
        """"Returns the height of the sensor."""
        return self._height
    @property
    def center(self) -> np.ndarray:
        """"Returns the center of the sensor."""
        return self._center
    @property
    def intrinsics(self) -> np.ndarray:
        """Obtain an array of the intrinsics of this camera model.
        Returns:
            np.ndarray: an array of intrinsics. The ordering is "cx, cy, width, height, bw_poly".
                dtype is np.float32.
        """
        return self._intrinsics
    def __str__(self):
        """Returns a string representation of this object."""
        return (
            f"FTheta camera model:\n\t{self.bw_poly}\n\t"
            f"center={self._center}\n\twidth={self._width}\n\theight={self._height}\n\t"
            f"h_fov={np.degrees(self._horizontal_fov)}\n\tv_fov={np.degrees(self._vertical_fov)}"
        )
    def _update_calibrated_camera(self):
        """Updates the internals of this object after calulating various properties."""
        self._compute_fov()
        self._max_ray_angle = (self._max_angle).copy()
        is_fw_poly_slope_negative_in_domain = False
        ray_angle = (np.float32(self._max_ray_angle)).copy()
        deg2rad = np.pi / 180.0
        while ray_angle >= np.float32(0.0):
            temp_dval = self._fw_poly.deriv()(self._max_ray_angle).item()
            if temp_dval < 0:
                is_fw_poly_slope_negative_in_domain = True
            ray_angle -= deg2rad * np.float32(1.0)
        if is_fw_poly_slope_negative_in_domain:
            ray_angle = (np.float32(self._max_ray_angle)).copy()
            while ray_angle >= np.float32(0.0):
                ray_angle -= deg2rad * np.float32(1.0)
            raise ArithmeticError(
                "FThetaCamera: derivative of distortion within image interior is negative"
            )
        # Evaluate the forward polynomial at point (self._max_ray_angle, 0)
        # Also evaluate its derivative at the same point
        val = self._fw_poly(self._max_ray_angle).item()
        dval = self._fw_poly.deriv()(self._max_ray_angle).item()
        if dval < 0:
            raise ArithmeticError(
                "FThetaCamera: derivative of distortion at edge of image is negative"
            )
        self._max_ray_distortion = np.asarray([val, dval], dtype=np.float32)
    def _compute_fw_poly(self):
        """Computes the forward polynomial for this camera.
        This function is a replication of the logic in the following file from the DW repo:
        src/dw/calibration/cameramodel/CameraModels.cpp
        """
        def get_max_value(p0, p1):
            return np.linalg.norm(
                np.asarray([p0, p1], dtype=self._center.dtype) - self._center
            )
        max_value = 0.0
        size = (self._width, self._height)
        value = get_max_value(0.0, 0.0)
        max_value = max(max_value, value)
        value = get_max_value(0.0, size[1])
        max_value = max(max_value, value)
        value = get_max_value(size[0], 0.0)
        max_value = max(max_value, value)
        value = get_max_value(size[0], size[1])
        max_value = max(max_value, value)
        SAMPLE_COUNT = 500
        samples_x = []
        samples_b = []
        step = max_value / SAMPLE_COUNT
        x = step
        for _ in range(0, SAMPLE_COUNT):
            p = np.asarray([self._center[0] + x, self._center[1]], dtype=np.float32)
            ray, _ = self.pixel2ray(p)
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
            return b + x * (x1 + x * (x2 + x * (x3 + x * x4)))
        coeffs, _ = curve_fit(
            f,
            x,
            y,
            bounds=(
                [0, -np.inf, -np.inf, -np.inf, -np.inf],
                [np.finfo(np.float64).eps, np.inf, np.inf, np.inf, np.inf],
            ),
        )
        # Return the polynomial and hardcode the bias value to 0
        return Polynomial(
            [np.float32(val) if i > 0 else 0 for i, val in enumerate(coeffs)]
        )
    def pixel2ray(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Backproject 2D pixels into 3D rays.
        Args:
            x (np.ndarray): the pixels to backproject. Size of (n_points, 2), where the first
                column contains the `x` values, and the second column contains the `y` values.
        Returns:
            rays (np.ndarray): the backprojected 3D rays. Size of (n_points, 3).
            valid (np.ndarray): bool flag indicating the validity of each backprojected pixel.
        """
        # Make sure x is n x 2
        if np.ndim(x) == 1:
            x = x[np.newaxis, :]
        # Fix the type
        x = x.astype(np.float32)
        xd = x - self._center
        xd_norm = np.linalg.norm(xd, axis=1, keepdims=True)
        alpha = self.bw_poly(xd_norm)
        sin_alpha = np.sin(alpha)
        rx = sin_alpha * xd[:, 0:1] / xd_norm
        ry = sin_alpha * xd[:, 1:] / xd_norm
        rz = np.cos(alpha)
        rays = np.hstack((rx, ry, rz))
        # special case: ray is perpendicular to image plane normal
        valid = (xd_norm > np.finfo(np.float32).eps).squeeze()
        rays[~valid, :] = (0, 0, 1)  # This is what DW sets these rays to
        # note:
        # if constant coefficient of bwPoly is non-zero,
        # the resulting ray might not be normalized.
        return rays, valid
    def ray2pixel(self, rays: np.ndarray) -> np.ndarray:
        """Project 3D rays to 2D pixel coordinates.
        Args:
            rays (np.ndarray): the rays.
        Returns:
            result (np.ndarray): the projected pixel coordinates.
        """
        # Make sure the input shape is (n_points, 3)
        if np.ndim(rays) == 1:
            rays = rays[np.newaxis, :]
        # Fix the type
        rays = rays.astype(np.float32)
        xy_norm = np.linalg.norm(rays[:, :2], axis=1, keepdims=True)
        cos_alpha = rays[:, 2:] / np.linalg.norm(rays, axis=1, keepdims=True)
        alpha = np.empty_like(cos_alpha)
        cos_alpha_condition = np.logical_and(
            cos_alpha > np.float32(-1.0), cos_alpha < np.float32(1.0)
        ).squeeze()
        alpha[cos_alpha_condition] = np.arccos(cos_alpha[cos_alpha_condition])
        alpha[~cos_alpha_condition] = xy_norm[~cos_alpha_condition]
        delta = np.empty_like(cos_alpha)
        alpha_cond = alpha <= self._max_ray_angle
        delta[alpha_cond] = self._fw_poly(alpha[alpha_cond])
        # For outside the model (which need to do linear extrapolation)
        delta[~alpha_cond] = (
            self._max_ray_distortion[0]
            + (alpha[~alpha_cond] - self._max_ray_angle) * self._max_ray_distortion[1]
        )
        # Determine the bad points with a norm of zero, and avoid division by zero
        bad_norm = xy_norm <= 0
        xy_norm[bad_norm] = 1
        delta[bad_norm] = 0
        # compute pixel relative to center
        scale = delta / xy_norm
        pixel = scale * rays
        # Handle the edge cases (ray along image plane normal)
        edge_case_cond = (xy_norm <= np.float32(0.0)).squeeze()
        pixel[edge_case_cond, :] = rays[edge_case_cond, :]
        result = pixel[:, :2] + self._center
        return result
    def _get_pixel_fov(self, pt: np.ndarray) -> float:
        """Gets the FOV for a given point. Used internally for FOV computation.
        Args:
            pt (np.ndarray): 2D pixel.
        Returns:
            fov (float): the FOV of the pixel.
        """
        ray, _ = self.pixel2ray(pt)
        fov = np.arctan2(np.linalg.norm(ray[:, :2], axis=1), ray[:, 2])
        return fov
    def _compute_fov(self):
        """Computes the FOV of this camera model."""
        max_x = self._width - 1
        max_y = self._height - 1
        point_left = np.asarray([0, self._center[1]], dtype=np.float32)
        point_right = np.asarray([max_x, self._center[1]], dtype=np.float32)
        point_top = np.asarray([self._center[0], 0], dtype=np.float32)
        point_bottom = np.asarray([self._center[0], max_y], dtype=np.float32)
        fov_left = self._get_pixel_fov(point_left)
        fov_right = self._get_pixel_fov(point_right)
        fov_top = self._get_pixel_fov(point_top)
        fov_bottom = self._get_pixel_fov(point_bottom)
        self._vertical_fov = fov_top + fov_bottom
        self._horizontal_fov = fov_left + fov_right
        self._compute_max_angle()
    def _compute_max_angle(self):
        """Computes the maximum ray angle for this camera."""
        max_x = self._width - 1
        max_y = self._height - 1
        p = np.asarray(
            [[0, 0], [max_x, 0], [0, max_y], [max_x, max_y]], dtype=np.float32
        )
        self._max_angle = max(
            max(self._get_pixel_fov(p[0, ...]), self._get_pixel_fov(p[1, ...])),
            max(self._get_pixel_fov(p[2, ...]), self._get_pixel_fov(p[3, ...])),
        )
    def is_ray_inside_fov(self, ray: np.ndarray) -> bool:
        """Determines whether a given ray is inside the FOV of this camera.
        Args:
            ray (np.ndarray): the 3D ray.
        Returns:
            bool: whether the ray is inside the FOV.
        """
        if np.ndim(ray) == 1:
            ray = ray[np.newaxis, :]
        ray_angle = np.arctan2(np.linalg.norm(ray[:, :2], axis=1), ray[:, 2])
        return ray_angle <= self._max_angle

def rectify(img, model, ph_model: IdealPinholeCamera):
    """Apply rectification to the frame via backward sampling.
    Args:
        ph_model: pinhole camera model used to rectify the image.
    Returns:
        A copy of the CameraFrame object with updated `data`, `model`,
          and `car_mask` attributes.
    """
    # we assume it's an FTheta camera model
    assert isinstance(model, FThetaCamera)
    ftheta_model: FThetaCamera = model
    # compute rays through each pixel in pinhole image
    ii, jj = np.meshgrid(
        np.arange(ph_model.height), np.arange(ph_model.width), indexing="ij"
    )
    ph_coords = np.stack((jj, ii), axis=-1)  # (H, W, 2)
    ph_coords = ph_coords.reshape(-1, 2)  # (H*W, 2)
    ph_rays = ph_model.pixel2ray(ph_coords)
    # project rays into ftheta image
    ftheta_coords = ftheta_model.ray2pixel(ph_rays)
    # sample ftheta RGB and car mask
    map_xy = ftheta_coords.reshape((ph_model.height, ph_model.width, 2))
    sampled_rgb = cv2.remap(
        img, map_xy[..., 0], map_xy[..., 1], interpolation=cv2.INTER_LINEAR
    )
    # return new camera frame container
    """
    rect = CameraFrame(
        timestamp=None,
        frame_number=None,
        sensor_name=None,
        data=sampled_rgb,
        rig=None,
        model=ph_model,
    )
    """
    return sampled_rgb

def ph_pixel2ray(pixels: np.ndarray, k_inv) -> np.ndarray:
    """Backproject 2D pixels into 3D rays.
    Args:
        pixels (np.ndarray): the pixels to backproject. Size of (n_points, 2), where the first
            column contains the `x` values, and the second column contains the `y` values.
    Returns:
        rays (np.ndarray): the backprojected 3D rays.
    """
    if np.ndim(pixels) == 1:
        pixels = pixels[np.newaxis, :]
    pixels = pixels.astype(np.float32)
    # Add the third component of ones
    pixels = np.c_[pixels, np.ones((pixels.shape[0], 1), dtype=np.float32)]
    rays = np.matmul(k_inv, pixels.T).T
    # Normalize the rays
    norm = np.linalg.norm(rays, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return rays / norm

def ftheta_pixel2ray(x: np.ndarray, center, bw_poly) -> Tuple[np.ndarray, np.ndarray]:
    """Backproject 2D pixels into 3D rays.
    Args:
        x (np.ndarray): the pixels to backproject. Size of (n_points, 2), where the first
            column contains the `x` values, and the second column contains the `y` values.
    Returns:
        rays (np.ndarray): the backprojected 3D rays. Size of (n_points, 3).
        valid (np.ndarray): bool flag indicating the validity of each backprojected pixel.
    """
    # Make sure x is n x 2
    if np.ndim(x) == 1:
        x = x[np.newaxis, :]
    # Fix the type
    x = x.astype(np.float32)
    xd = x - center
    xd_norm = np.linalg.norm(xd, axis=1, keepdims=True)
    alpha = bw_poly(xd_norm)
    sin_alpha = np.sin(alpha)
    rx = sin_alpha * xd[:, 0:1] / xd_norm
    ry = sin_alpha * xd[:, 1:] / xd_norm
    rz = np.cos(alpha)
    rays = np.hstack((rx, ry, rz))
    # special case: ray is perpendicular to image plane normal
    valid = (xd_norm > np.finfo(np.float32).eps).squeeze()
    rays[~valid, :] = (0, 0, 1)  # This is what DW sets these rays to
    # note:
    # if constant coefficient of bwPoly is non-zero,
    # the resulting ray might not be normalized.
    return rays, valid

def ftheta_ray2pixel(rays: np.ndarray, max_ray_angle, max_ray_distortion, center, fw_poly) -> np.ndarray:
    """Project 3D rays to 2D pixel coordinates.
    Args:
        rays (np.ndarray): the rays.
    Returns:
        result (np.ndarray): the projected pixel coordinates.
    """
    # Make sure the input shape is (n_points, 3)
    if np.ndim(rays) == 1:
        rays = rays[np.newaxis, :]
    # Fix the type
    rays = rays.astype(np.float32)
    xy_norm = np.linalg.norm(rays[:, :2], axis=1, keepdims=True)
    cos_alpha = rays[:, 2:] / np.linalg.norm(rays, axis=1, keepdims=True)
    alpha = np.empty_like(cos_alpha)
    cos_alpha_condition = np.logical_and(
        cos_alpha > np.float32(-1.0), cos_alpha < np.float32(1.0)
    ).squeeze()
    alpha[cos_alpha_condition] = np.arccos(cos_alpha[cos_alpha_condition])
    alpha[~cos_alpha_condition] = xy_norm[~cos_alpha_condition]
    delta = np.empty_like(cos_alpha)
    alpha_cond = alpha <= max_ray_angle
    delta[alpha_cond] = fw_poly(alpha[alpha_cond])
    # For outside the model (which need to do linear extrapolation)
    delta[~alpha_cond] = (
        max_ray_distortion[0]
        + (alpha[~alpha_cond] - max_ray_angle) * max_ray_distortion[1]
    )
    # Determine the bad points with a norm of zero, and avoid division by zero
    bad_norm = xy_norm <= 0
    xy_norm[bad_norm] = 1
    delta[bad_norm] = 0
    # compute pixel relative to center
    scale = delta / xy_norm
    pixel = scale * rays
    # Handle the edge cases (ray along image plane normal)
    edge_case_cond = (xy_norm <= np.float32(0.0)).squeeze()
    pixel[edge_case_cond, :] = rays[edge_case_cond, :]
    result = pixel[:, :2] + center
    return result

def compute_fw_poly(height, width, center):
    """Computes the forward polynomial for this camera.
    This function is a replication of the logic in the following file from the DW repo:
    src/dw/calibration/cameramodel/CameraModels.cpp
    """
    def get_max_value(p0, p1):
        return np.linalg.norm(
            np.asarray([p0, p1], dtype=center.dtype) - center
        )
    max_value = 0.0
    size = (width, height)
    value = get_max_value(0.0, 0.0)
    max_value = max(max_value, value)
    value = get_max_value(0.0, size[1])
    max_value = max(max_value, value)
    value = get_max_value(size[0], 0.0)
    max_value = max(max_value, value)
    value = get_max_value(size[0], size[1])
    max_value = max(max_value, value)
    SAMPLE_COUNT = 500
    samples_x = []
    samples_b = []
    step = max_value / SAMPLE_COUNT
    x = step
    for _ in range(0, SAMPLE_COUNT):
        p = np.asarray([center[0] + x, center[1]], dtype=np.float32)
        ray, _ = ftheta_pixel2ray(p)
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
        return b + x * (x1 + x * (x2 + x * (x3 + x * x4)))
    coeffs, _ = curve_fit(
        f,
        x,
        y,
        bounds=(
            [0, -np.inf, -np.inf, -np.inf, -np.inf],
            [np.finfo(np.float64).eps, np.inf, np.inf, np.inf, np.inf],
        ),
    )
    # Return the polynomial and hardcode the bias value to 0
    return Polynomial(
        [np.float32(val) if i > 0 else 0 for i, val in enumerate(coeffs)]
    )

def rectify_fast(img, height, width, cx, cy, fx, fy, bw_poly):
    """Apply rectification to the frame via backward sampling.
    Args:
        ph_model: pinhole camera model used to rectify the image.
    Returns:
        A copy of the CameraFrame object with updated `data`, `model`,
          and `car_mask` attributes.
    """
    # ASSUME PH AND FTHETA SHARE THE SAME CENTER
    center = np.asarray([cx, cy], dtype=np.float32)
    k_inv = np.asarray(
        [
            [1.0 / fx, 0, -cx / fx],
            [0, 1.0 / fy, -cy / fy],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    # compute rays through each pixel in pinhole image
    ii, jj = np.meshgrid(
        np.arange(height), np.arange(width), indexing="ij"
    )
    ph_coords = np.stack((jj, ii), axis=-1)  # (H, W, 2)
    ph_coords = ph_coords.reshape(-1, 2)  # (H*W, 2)
    ph_rays = ph_pixel2ray(ph_coords, k_inv)
    # project rays into ftheta image
    ftheta_coords = ftheta_ray2pixel(ph_rays)
    # sample ftheta RGB and car mask
    map_xy = ftheta_coords.reshape((height, width, 2))
    sampled_rgb = cv2.remap(
        img, map_xy[..., 0], map_xy[..., 1], interpolation=cv2.INTER_LINEAR
    )
    return sampled_rgb
