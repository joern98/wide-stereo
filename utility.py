import datetime
import os
from dataclasses import dataclass
from typing import Tuple

import cv2 as cv
import numpy as np

import open3d as o3d
import pyrealsense2 as rs

from device_utility.utils import get_stereo_extrinsic


@dataclass()
class CameraParameters:
    left_intrinsics: rs.intrinsics
    right_intrinsics: rs.intrinsics
    left_pinhole_intrinsics: o3d.camera.PinholeCameraIntrinsic
    right_pinhole_intrinsics: o3d.camera.PinholeCameraIntrinsic
    left_stereo_extrinsic: rs.extrinsics
    right_stereo_extrinsic: rs.extrinsics
    image_size: Tuple[int, int]


def save_point_cloud(pc: o3d.geometry.PointCloud):
    POINT_CLOUD_SAVE_DIR = os.path.join("PointClouds")
    filename = f"PointCloud_{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S-%f')}.ply"
    filepath = os.path.join(POINT_CLOUD_SAVE_DIR, filename)
    o3d.io.write_point_cloud(filepath, pc)
    print(f"Point-Cloud saved to {filepath}")


def get_camera_parameters(device_pair):
    left_intrinsic: rs.intrinsics = device_pair.left.pipeline_profile.get_stream(
        rs.stream.depth).as_video_stream_profile().get_intrinsics()
    right_intrinsic: rs.intrinsics = device_pair.right.pipeline_profile.get_stream(
        rs.stream.depth).as_video_stream_profile().get_intrinsics()

    left_pinhole_intrinsics = intrinsics_to_o3d_pinhole_intrinsic(left_intrinsic)
    right_pinhole_intrinsics = intrinsics_to_o3d_pinhole_intrinsic(right_intrinsic)

    left_stereo_extrinsic = get_stereo_extrinsic(device_pair.left.pipeline_profile)
    right_stereo_extrinsic = get_stereo_extrinsic(device_pair.right.pipeline_profile)
    return CameraParameters(left_intrinsic, right_intrinsic,
                            left_pinhole_intrinsics, right_pinhole_intrinsics,
                            left_stereo_extrinsic, right_stereo_extrinsic,
                            (left_intrinsic.width, left_intrinsic.height))


def intrinsics_to_o3d_pinhole_intrinsic(intrinsic: rs.intrinsics):
    return o3d.camera.PinholeCameraIntrinsic(width=intrinsic.width,
                                             height=intrinsic.height,
                                             cx=intrinsic.ppx,
                                             cy=intrinsic.ppy,
                                             fx=intrinsic.fx,
                                             fy=intrinsic.fy)


def save_screenshot(image: np.ndarray, mouse_position=(0, 0)):
    SCREENSHOT_SAVE_DIR = os.path.join("Screenshots")
    filename = f"Screenshot_{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S-%f')}.png"
    # add marker at mouse position to the saved image
    cv.drawMarker(image, mouse_position, [0, 0, 0], cv.MARKER_SQUARE, markerSize=6, thickness=1)
    filepath = os.path.join(SCREENSHOT_SAVE_DIR, filename)
    cv.imwrite(filepath, image)
    print(f"Screenshot saved to {filepath}")
