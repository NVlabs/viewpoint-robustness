"""
Copyright (C) 2023 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/NVlabs/viewpoint-robustness
Authors: Tzofi Klinghoffer, Jonah Philion, Wenzheng Chen, Or Litany, Zan Gojcic, Jungseock Joo, Ramesh Raskar, Sanja Fidler, Jose M. Alvarez
"""

"""Type definitions for Obstacle3D."""
from typing import Dict, List, NamedTuple, Tuple
class TopDownLidarGT(NamedTuple):
    """Collection of top-down 2D lidar GT features transformed into the rig frame."""
    # Lidar label IDs (used for association between camera/lidar GT labels).
    label_ids: List[str]
    # label names (automobile, etc.).
    label_names: List[str]
    # 2-level list of x(col) and y(row) coordinates (in the rig frame).
    # The outer list is over objects, the inner list is over vertices.
    vertices: List[List[Tuple[float, float]]]
    # The number of vertices in the each list inside `vertices`
    vertex_count: List[int]
class Center(NamedTuple):
    """Describes a Center coordinate.
    contains center x, y, z
    """
    x: float
    y: float
    z: float
class Orientation(NamedTuple):
    """Describes an Orientation coordinate.
    Contains roll, pitch, and yaw.
    """
    yaw: float
    pitch: float
    roll: float
class Dimension(NamedTuple):
    """Describes a Dimension coordinate.
    This contains dimension along x, y, z.
    """
    x: float
    y: float
    z: float
class BBox2D(NamedTuple):
    """Describes a single 2D bounding box.
    x1, y1, x2, y2 correspond to x_min, y_min, x_max, y_max
    """
    x1: float
    y1: float
    x2: float
    y2: float
    @classmethod
    def from_ilf(cls, box2d):
        """
        Parses the bounding box 2D from the dataset.
        Returns:
            BBox2D: an instance of BBox2D class.
        """
        return cls(x1=box2d[0][0], y1=box2d[0][1], x2=box2d[1][0], y2=box2d[1][1])
class Cuboid3D(NamedTuple):
    """Describes a single 3D cuboid.
    contains Center, Dimension, and Orientation
    classes.
    """
    center: Center
    orientation: Orientation
    dimension: Dimension
    @classmethod
    def from_ilf(cls, cuboid3d):
        """
        Parses the cuboid3d structure from the dataset.
        Converts from a json structure to python
        Cuboid3D structure.
        Returns:
            Cuboid3D: an instance of Cuboid3D class.
        """
        center = Center(cuboid3d.center.x, cuboid3d.center.y, cuboid3d.center.z)
        orientation = Orientation(
            cuboid3d.orientation.x, cuboid3d.orientation.y, cuboid3d.orientation.z
        )
        dimension = Dimension(
            cuboid3d.dimensions.x, cuboid3d.dimensions.y, cuboid3d.dimensions.z
        )
        return cls(center=center, orientation=orientation, dimension=dimension)
class Cuboid3DGT(NamedTuple):
    """Collection of 2D bounding box and 3D cuboid GT features."""
    # Lidar label IDs: Used for 2D bbox and 3D Cuboid association.
    label_ids: List[str]
    # automobile, etc.
    label_names: List[str]
    # list of 2D bounding boxes from multiple cameras
    bbox2ds: List[Dict[str, BBox2D]]
    # list of 3D cuboids
    cuboid3ds: List[Cuboid3D]
    # visibility of 3D Cuboids
    # Values range between [0,1] that indicates the level of occlusion
    # If cuboid is occluded it will have a value closer to 0
    # If cuboid is not occluded it will have a value near 1
    # defaults to -1.0 if not present when feature parsing which means ignore
    visibility: List[float]
    # Per-camera visibility attribute. One vis float per cam if the
    # cuboid is visible in that camera. Jira AMP-454 is task to track
    # a merging of this attribute with the above visibility[List[float]]
    visibility_per_camera: List[Dict[str, float]]

