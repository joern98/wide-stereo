import math
from os import path
from typing import Tuple, Sequence

import cv2 as cv
import numpy as np
import argparse
import json

from line_profiler_pycharm import profile

from plane_sweep_ext import compute_consistency_image

from device_utility.camera_calibration import load_calibration_from_file

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
@profile
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
    g_std = np.sqrt(np.sum((g-g_avg)**2))
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

    for i in range(n_planes):
        z = z_min + i * z_step
        print(f"Plane at z={z}")
        _L = [images[0]]
        for j in range(1, len(images)):
            # L[0] is not warped, projection would only scale the image
            H = compute_homography(k_rt[0], k_rt[j], z)
            projected = cv.warpPerspective(images[j], H, image_size, flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR)
            _L.append(projected)
        for m in range(len(_L)):
            cv.imshow(f"Camera {m}", _L[m])

        cv.waitKey(1)
        # TODO implement with Cython

        ref = _L[0]
        src = np.asarray(_L[1:])

        compute_consistency_image(ref, src, cost_volume[i], 3)

        v = np.ma.masked_less(cost_volume[i], 0)
        cv.imshow("cost_volume", v)
        cv.waitKey()


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
    plane_sweep(images, transforms, camera_parameters["image_size"], z_min=1.43, z_max=2.6, z_step=0.1)

    return 0
    MOUSE_X, MOUSE_Y = 0, 0

    def on_mouse(event, x, y, flags, user_data):
        nonlocal MOUSE_X, MOUSE_Y
        if event == cv.EVENT_MOUSEMOVE:
            MOUSE_X, MOUSE_Y = x, y

    run = True
    while run:
        key = cv.waitKey(1)
        if key == 27:  # ESCAPE
            run = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Path to the directory created by capture.py")
    args = parser.parse_args()
    main(args)
