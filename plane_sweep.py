from os import path
import cv2 as cv
import numpy as np
import argparse
import json

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
    return np.asmatrix(k, dtype=np.float32)


def main(args):
    calibration_result, camera_parameters, \
        left_ir_1, left_ir_2, right_ir_1, right_ir_2, \
        left_native_depth, right_native_depth = load_data(args.directory)

    d = 2.45
    # normal direction has to be negative z
    n = np.asmatrix([0, 0, -1])

    KR = get_camera_matrix_from_dict(camera_parameters["right_intrinsics"])
    KL = get_camera_matrix_from_dict(camera_parameters["left_intrinsics"])
    R = np.asmatrix(calibration_result.R_14[:3, :3])
    t = np.asmatrix(calibration_result.R_14[:3, 3:4]).reshape(1, 3)
    tn = t.T @ n
    H2 = KR @ (R - tn / d) @ KL.I
    cv.imshow(WINDOW_LEFT_IR_1, left_ir_1)
    cv.imshow(WINDOW_RIGHT_IR_2, right_ir_2)

    # this works as well, but we need to use the inverse, i.e. instead of p1=H*p0 we have p0=H.I*p1
    # setting the flag WARP_INVERSE_MAP to directly use the specified homography
    warped_right = cv.warpPerspective(right_ir_2, H2, (1280, 720), flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR)

    MOUSE_X, MOUSE_Y = 0, 0

    def on_mouse(event, x, y, flags, user_data):
        nonlocal MOUSE_X, MOUSE_Y
        if event == cv.EVENT_MOUSEMOVE:
            MOUSE_X, MOUSE_Y = x, y

    cv.setMouseCallback(WINDOW_LEFT_IR_1, on_mouse)

    run = True
    while run:
        p2 = (H2 @ [MOUSE_X, MOUSE_Y, 1]).reshape(-1, 3).astype(int)
        img = cv.copyTo(right_ir_2, None)
        cv.drawMarker(img, (p2[0, 0], p2[0, 1]), (1, 1, 1))
        cv.imshow(WINDOW_RIGHT_IR_2, img)

        warped_right_marker = cv.copyTo(warped_right, None)
        cv.drawMarker(warped_right_marker, (MOUSE_X, MOUSE_Y), (1, 1, 1))
        cv.imshow(WINDOW_RIGHT_IR_1, warped_right_marker)
        key = cv.waitKey(1)
        if key == 27:  # ESCAPE
            run = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Path to the directory created by capture.py")
    args = parser.parse_args()
    main(args)
