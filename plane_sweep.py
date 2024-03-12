import math
import os
import time
from datetime import datetime
from os import path
from typing import Tuple, Sequence

import cv2 as cv
import numpy as np
import argparse
import json
import open3d as o3d

from line_profiler_pycharm import profile

# imported .pyd, ignore error
from plane_sweep_ext import compute_consistency_image
from scipy.interpolate import interp1d

from device_utility.camera_calibration import load_calibration_from_file
from utility import save_screenshot, save_point_cloud

WINDOW_LEFT_IR_1 = "left IR 1"
WINDOW_LEFT_IR_2 = "left IR 2"
WINDOW_RIGHT_IR_1 = "right IR 1"
WINDOW_RIGHT_IR_2 = "right IR 2"


def load_data(directory: str):
    if directory is None or directory == "":
        raise Exception("Directory not given")
    calibration = load_calibration_from_file(path.join(directory, "Calibration.npy"))
    with open(path.join(directory, "CameraParameters.json")) as f:
        camera_parameters = json.load(f)
    left_ir_1 = cv.imread(path.join(directory, "left_ir_1.png"))
    left_ir_2 = cv.imread(path.join(directory, "left_ir_2.png"))
    right_ir_1 = cv.imread(path.join(directory, "right_ir_1.png"))
    right_ir_2 = cv.imread(path.join(directory, "right_ir_2.png"))
    left_depth = np.load(path.join(directory, "left_depth_raw.npy"))
    right_depth = np.load(path.join(directory, "right_depth_raw.npy"))

    return calibration, camera_parameters, left_ir_1, left_ir_2, right_ir_1, right_ir_2, left_depth, right_depth


def get_camera_matrix_from_dict(intrinsic_dict):
    k = [[intrinsic_dict["fx"], 0, intrinsic_dict["ppx"]],
         [0, intrinsic_dict["fy"], intrinsic_dict["ppy"]],
         [0, 0, 1]]
    return np.asanyarray(k, dtype=np.float32).reshape(3, 3)


#       KR = get_camera_matrix_from_dict(camera_parameters["right_intrinsics"])
#     KL = get_camera_matrix_from_dict(camera_parameters["left_intrinsics"])
#     R = np.asmatrix(calibration_result.R_14[:3, :3])
#     t = np.asmatrix(calibration_result.R_14[:3, 3:4]).reshape(1, 3)
#     tn = t.T @ n
#     H2 = KR @ (R - tn / d) @ KL.I
def compute_homography(k_rt0: Tuple[np.ndarray, np.ndarray, np.ndarray],
                       k_rt1: Tuple[np.ndarray, np.ndarray, np.ndarray],
                       z: float) -> np.ndarray:
    n = np.asarray([0, 0, -1]).reshape(3, 1)
    K0_I = np.linalg.inv(k_rt0[0])
    K1 = k_rt1[0]
    R = k_rt1[1]
    t = k_rt1[2]
    tn = t @ n.T
    H = K1 @ (R - tn / z) @ K0_I
    return H


def compute_sad(L0, L1, u, v, window_size):
    _radius = window_size // 2
    if u - _radius < 0 or v - _radius < 0 or u + _radius >= L0.shape[0] or v + _radius >= L0.shape[1]:
        return 0
    L0_W = L0[u - _radius: u + _radius + 1, v - _radius: v + _radius + 1]
    L1_W = L1[u - _radius: u + _radius + 1, v - _radius: v + _radius + 1]
    sum = np.sum(np.abs(L0_W - L1_W))
    return sum


def compute_ssd(L0, L1, u, v, window_size):
    _radius = window_size // 2
    if u - _radius < 0 or v - _radius < 0 or u + _radius >= L0.shape[0] or v + _radius >= L0.shape[1]:
        return 0
    L0_W = L0[u - _radius: u + _radius + 1, v - _radius: v + _radius + 1]
    L1_W = L1[u - _radius: u + _radius + 1, v - _radius: v + _radius + 1]
    sum = np.sum(np.square(L0_W - L1_W))
    return sum


# Normalized Cross-Correlation, (Furukawa, Hernandez: Multi-View Stereo, p.23), (Szeliski: Computer Vision A&A, p. 448)
def compute_ncc(L0, L1, u, v, window_size=3):
    _radius = window_size // 2
    if u - _radius < 0 or v - _radius < 0 or u + _radius >= L0.shape[0] or v + _radius >= L0.shape[1]:
        return 0
    f = L0[u - _radius: u + _radius + 1, v - _radius: v + _radius + 1]
    g = L1[u - _radius: u + _radius + 1, v - _radius: v + _radius + 1]

    f_avg = np.mean(f)
    g_avg = np.mean(g)
    n = window_size * window_size

    # this is not actually the standard deviation, but leads to correct results and is given this way by Szeliski, p.448
    f_std = np.sqrt(np.sum((f - f_avg) ** 2))
    g_std = np.sqrt(np.sum((g - g_avg) ** 2))
    fg_std = f_std * g_std
    if fg_std == 0:
        return 0

    p = np.sum((f - f_avg) * (g - g_avg))
    ncc = p / fg_std
    return ncc


def plane_sweep(images: [cv.Mat | np.ndarray | cv.UMat], k_rt: [Tuple[np.ndarray, np.ndarray, np.ndarray]],
                image_size: Sequence[int],
                z_min: float, z_max: float, z_step: float):
    """
    Perform the basic plane sweeping algorithm

    :param images: Array of images
    :param k_rt: Array of Tuples (K, R, t)
    :param image_size:
    :param z_min:
    :param z_max:
    :param z_step:
    :return:
    """
    n_planes = math.floor((z_max - z_min) / z_step) + 1
    cost_volume = np.zeros((n_planes, image_size[1], image_size[0]), dtype=np.float32)

    # Fill cost volume
    for i in range(n_planes):
        z = z_min + i * z_step
        print(f"Plane at z={z}")
        _L = [images[0]]
        for j in range(1, len(images)):
            # L[0] is not warped, projection would only scale the image
            H = compute_homography(k_rt[0], k_rt[j], z)
            projected = cv.warpPerspective(images[j], H, image_size, flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR)
            _L.append(projected)
        # for m in range(len(_L)):
        #     cv.imshow(f"Camera {m}", _L[m])

        # cv.waitKey(1)
        # TODO implement with Cython

        ref = _L[0]
        src = np.asarray(_L[1:])

        start = time.perf_counter_ns()
        compute_consistency_image(ref, src, cost_volume[i], 7)
        print(f"Consistency computation took {(time.perf_counter_ns() - start) / 1000000} ms")
        # v = (cost_volume[i] + 1.0) / 2
        # cv.imshow("cost_volume", v)
        # cv.waitKey(1)

    # find depth
    # np.argmax returns the index of max element across axis
    max_idx = np.argmax(cost_volume, axis=0)
    depth = z_min + max_idx * z_step

    # Uniqueness Ratio to reduce noise
    # for this to work properly, I would need to analyze all planes and see, how unique the maximum is over the whole domain
    # in the current form it also filters out areas where a valid depth would be found (room wall) but the maximum is not "sharp"
    # sorted_idx = np.argsort(cost_volume, axis=0)
    # sorted_cost = np.take_along_axis(cost_volume, sorted_idx, axis=0)
    # cost_ratio = np.abs(np.divide(sorted_cost[-2], sorted_cost[-1]))
    # max_ratio = 1 - 0.005
    # cost_ratio_threshold = np.where(cost_ratio < max_ratio, cost_ratio, -1)
    # depth = np.where(cost_ratio_threshold != -1, z_min + sorted_idx[-1] * z_step, 0)

    return depth


def compute_transforms(calibration_result, camera_parameters) -> [Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    R_01 = np.asarray(camera_parameters["left_stereo_extrinsics"]["r"]).reshape(3, 3)
    t_01 = np.asarray(camera_parameters["left_stereo_extrinsics"]["t"]).reshape(3, 1)

    R_12 = np.asarray(calibration_result.R)
    t_12 = np.asarray(calibration_result.T)

    R_23 = np.asarray(camera_parameters["right_stereo_extrinsics"]["r"]).reshape(3, 3)
    t_23 = np.asarray(camera_parameters["right_stereo_extrinsics"]["t"]).reshape(3, 1)

    R_02 = R_12 @ R_01
    t_02 = R_12 @ t_01 + t_12

    R_03 = R_23 @ R_02
    t_03 = R_23 @ R_12 @ t_01 + R_23 @ t_12 + t_23

    K0 = K1 = get_camera_matrix_from_dict(camera_parameters["left_intrinsics"])
    K2 = K3 = get_camera_matrix_from_dict(camera_parameters["right_intrinsics"])

    R_00 = np.eye(3)
    t_00 = np.zeros((1, 3))

    m = [(K0, R_00, t_00),
         (K1, R_01, t_01),
         (K2, R_02, t_02),
         (K3, R_03, t_03)]
    return m


OUTPUT_DIRECTORY = None


def ensure_output_directory(root_directory):
    global OUTPUT_DIRECTORY
    if OUTPUT_DIRECTORY is None:
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        pathname = path.join(root_directory, f"Output_{timestamp}")
        os.mkdir(pathname)
        OUTPUT_DIRECTORY = pathname
    return OUTPUT_DIRECTORY


def depth_to_point_cloud(depth: np.ndarray, intrinsic, extrinsic,
                         colormap: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    if colormap is not None:
        depth_image = o3d.geometry.Image(depth)
        # OpenCV is BGR, Open3D expects RGB
        depth_colormap_image = o3d.geometry.Image(colormap)
        depth_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=depth_colormap_image,
                                                                              depth=depth_image,
                                                                              depth_trunc=20,
                                                                              depth_scale=1,
                                                                              convert_rgb_to_intensity=False)
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(depth_rgbd_image, intrinsic, extrinsic)
    else:
        depth_image = o3d.geometry.Image(depth)
        pc = o3d.geometry.PointCloud.create_from_depth_image(depth_image,
                                                             intrinsic,
                                                             extrinsic,
                                                             depth_scale=1,
                                                             depth_trunc=20)
    return pc


def create_pinhole_intrinsic_from_dict(intrinsic_dict, image_size):
    return o3d.camera.PinholeCameraIntrinsic(width=image_size[0],
                                             height=image_size[1],
                                             cx=intrinsic_dict["ppx"],
                                             cy=intrinsic_dict["ppy"],
                                             fx=intrinsic_dict["fx"],
                                             fy=intrinsic_dict["fy"])


def main(args):
    calibration_result, camera_parameters, \
        left_ir_1, left_ir_2, right_ir_1, right_ir_2, \
        left_native_depth, right_native_depth = load_data(args.directory)

    # why are these greyscale images 3-channel? extract the first channel to save memory for plane sweep
    images = [cv.extractChannel(left_ir_1, 0),
              cv.extractChannel(left_ir_2, 0),
              cv.extractChannel(right_ir_1, 0),
              cv.extractChannel(right_ir_2, 0)]
    transforms = compute_transforms(calibration_result, camera_parameters)
    depth = plane_sweep(images, transforms, camera_parameters["image_size"], z_min=0.5, z_max=4.0, z_step=0.025)
    # depth = plane_sweep(images[::3], transforms[::3], camera_parameters["image_size"], z_min=0.5, z_max=4.0, z_step=0.1)  # only use outer cameras

    cv.destroyAllWindows()
    m = interp1d((0, 8), (0, 255), bounds_error=False, fill_value=(0, 255))
    depth_colored = cv.applyColorMap(m(depth).astype(np.uint8), cv.COLORMAP_JET)
    cv.imshow("depth", depth_colored)

    # This is a hack we need for Open3D to be able to create the point cloud: scale depth to mm and convert to uint16
    depth_image = o3d.geometry.Image((depth * 1000).astype(np.uint16))
    intrinsic = create_pinhole_intrinsic_from_dict(camera_parameters["left_intrinsics"],
                                                   camera_parameters["image_size"])
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_image,
                                                                  intrinsic=intrinsic,
                                                                  extrinsic=np.eye(4),
                                                                  depth_scale=1000,
                                                                  depth_trunc=20)
    MOUSE_X, MOUSE_Y = 0, 0

    def on_mouse(event, x, y, flags, user_data):
        nonlocal MOUSE_X, MOUSE_Y
        if event == cv.EVENT_MOUSEMOVE:
            MOUSE_X, MOUSE_Y = x, y
            print(f"depth: {depth[MOUSE_Y, MOUSE_X]} m")

    cv.setMouseCallback("depth", on_mouse)
    key = cv.waitKey()

    def output_dir():
        return ensure_output_directory(args.directory)

    if key == ord('s'):  # ESCAPE
        cv.imwrite(path.join(output_dir(), "Depth_PlaneSweep.png"), depth_colored)
        save_point_cloud(point_cloud, "PointCloud_PlaneSweep", output_directory=output_dir())
    cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Path to the directory created by capture.py")
    args = parser.parse_args()
    main(args)
