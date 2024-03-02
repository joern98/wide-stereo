import argparse
import json
import os
from datetime import datetime

import cv2 as cv
import numpy as np
import pyrealsense2 as rs
from scipy.interpolate import interp1d

from device_utility.DeviceManager import DeviceManager
from device_utility.DevicePair import DevicePair
from device_utility.camera_calibration import load_calibration_from_file, stereo_rectify, run_camera_calibration, \
    write_calibration_to_file
from device_utility.utils import set_sensor_option
from utility import CameraParametersWithPinhole, get_camera_parameters

WINDOW_DEPTH_LEFT = "depth left"
WINDOW_DEPTH_RIGHT = "depth right"

WIDTH = 1280
HEIGHT = 720
FPS = 15

MAP_DEPTH_MM_TO_BYTE = interp1d([0, 13000], [0, 255], bounds_error=False, fill_value=(0, 255))


def set_device_options(device_pair: DevicePair):
    depth_sensor_left: rs.depth_sensor = device_pair.left.device.first_depth_sensor()
    depth_sensor_right: rs.depth_sensor = device_pair.right.device.first_depth_sensor()

    # enable emitter
    set_sensor_option(depth_sensor_left, rs.option.emitter_enabled, True)
    set_sensor_option(depth_sensor_right, rs.option.emitter_enabled, True)

    # set emitter to always on
    set_sensor_option(depth_sensor_left, rs.option.emitter_always_on, True)
    set_sensor_option(depth_sensor_right, rs.option.emitter_always_on, True)

    # set ir emitter to full power
    set_sensor_option(depth_sensor_left, rs.option.laser_power, 360)
    set_sensor_option(depth_sensor_right, rs.option.laser_power, 360)

    # turn off auto exposure
    set_sensor_option(depth_sensor_left, rs.option.enable_auto_exposure, 0)
    set_sensor_option(depth_sensor_right, rs.option.enable_auto_exposure, 0)


def write_images(left_frame: rs.composite_frame, right_frame: rs.composite_frame, parent_dir):
    left_depth = np.asanyarray(left_frame.get_depth_frame().get_data())
    right_depth = np.asanyarray(right_frame.get_depth_frame().get_data())

    left_depth_color = cv.applyColorMap(MAP_DEPTH_MM_TO_BYTE(left_depth).astype(np.uint8), cv.COLORMAP_JET)
    right_depth_color = cv.applyColorMap(MAP_DEPTH_MM_TO_BYTE(right_depth).astype(np.uint8), cv.COLORMAP_JET)

    left_ir_1 = np.asanyarray(left_frame.get_infrared_frame(1).get_data())
    left_ir_2 = np.asanyarray(left_frame.get_infrared_frame(2).get_data())

    right_ir_1 = np.asanyarray(right_frame.get_infrared_frame(1).get_data())
    right_ir_2 = np.asanyarray(right_frame.get_infrared_frame(2).get_data())

    join = os.path.join
    fn_left_depth_raw = join(parent_dir, f"left_depth_raw.npy")
    fn_right_depth_raw = join(parent_dir, f"right_depth_raw.npy")
    fn_left_depth_color = join(parent_dir, f"left_depth_color.png")
    fn_right_depth_color = join(parent_dir, f"right_depth_color.png")
    fn_left_ir_1 = join(parent_dir, f"left_ir_1.png")
    fn_left_ir_2 = join(parent_dir, f"left_ir_2.png")
    fn_right_ir_1 = join(parent_dir, f"right_ir_1.png")
    fn_right_ir_2 = join(parent_dir, f"right_ir_2.png")

    np.save(fn_left_depth_raw, left_depth)
    np.save(fn_right_depth_raw, right_depth)
    cv.imwrite(fn_left_depth_color, left_depth_color)
    cv.imwrite(fn_left_ir_1, left_ir_1)
    cv.imwrite(fn_left_ir_2, left_ir_2)
    cv.imwrite(fn_right_depth_color, right_depth_color)
    cv.imwrite(fn_right_ir_1, right_ir_1)
    cv.imwrite(fn_right_ir_2, right_ir_2)

    print(f"Written output files to {parent_dir}")


def write_camera_parameters(camera_parameters: CameraParametersWithPinhole, file_basename):
    r = {
        "left_intrinsics": {
            "fx": camera_parameters.left_intrinsics.fx,
            "fy": camera_parameters.left_intrinsics.fy,
            "coeffs": camera_parameters.left_intrinsics.coeffs,
            "ppx": camera_parameters.left_intrinsics.ppx,
            "ppy": camera_parameters.left_intrinsics.ppy,
        },
        "right_intrinsics": {
            "fx": camera_parameters.right_intrinsics.fx,
            "fy": camera_parameters.right_intrinsics.fy,
            "coeffs": camera_parameters.right_intrinsics.coeffs,
            "ppx": camera_parameters.right_intrinsics.ppx,
            "ppy": camera_parameters.right_intrinsics.ppy,
        },
        "left_stereo_extrinsics": {
            "t": camera_parameters.left_stereo_extrinsic.translation,
            "r": camera_parameters.left_stereo_extrinsic.rotation
        },
        "right_stereo_extrinsics": {
            "t": camera_parameters.right_stereo_extrinsic.translation,
            "r": camera_parameters.right_stereo_extrinsic.rotation
        },
        "image_size": camera_parameters.image_size
    }
    with open(file_basename + ".json", "x") as f:
        json.dump(r, f, indent=2)
        print(f"Written camera parameters to file: {file_basename + '.json'}")


def main(args):
    ctx = rs.context()
    device_manager = DeviceManager(ctx)
    device_pair = device_manager.create_device_pair_interactive()
    set_device_options(device_pair)

    if args.calibration:
        calibration_result = load_calibration_from_file(args.calibration)
        rectification_result = stereo_rectify(calibration_result.image_size, calibration_result)
    else:
        calibration_result, rectification_result = run_camera_calibration(device_pair)

    cv.namedWindow(WINDOW_DEPTH_LEFT)
    cv.namedWindow(WINDOW_DEPTH_RIGHT)

    device_pair.start(WIDTH, HEIGHT, FPS, streams=(rs.stream.infrared, rs.stream.depth))
    camera_parameters = get_camera_parameters(device_pair)
    run = True
    while run:
        left_frame, right_frame = device_pair.wait_for_frames()

        left_depth = np.asanyarray(left_frame.get_depth_frame().get_data())
        left_depth_color = cv.applyColorMap(MAP_DEPTH_MM_TO_BYTE(left_depth).astype(np.uint8), cv.COLORMAP_JET)
        right_depth = np.asanyarray(right_frame.get_depth_frame().get_data())
        right_depth_color = cv.applyColorMap(MAP_DEPTH_MM_TO_BYTE(right_depth).astype(np.uint8), cv.COLORMAP_JET)

        cv.imshow(WINDOW_DEPTH_LEFT, left_depth_color)
        cv.imshow(WINDOW_DEPTH_RIGHT, right_depth_color)

        key = cv.pollKey()
        if key == 27:  # ESCAPE
            run = False
        if key == 115:  # s
            timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
            parent_dir = f"CAPTURE_{timestamp}"
            os.mkdir(parent_dir)
            write_images(left_frame, right_frame, parent_dir)
            write_calibration_to_file(calibration_result, os.path.join(parent_dir, "Calibration"))
            write_camera_parameters(camera_parameters, os.path.join(parent_dir, "CameraParameters"))

    cv.destroyAllWindows()
    device_pair.stop()


def create_config(filename: str):
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, width=WIDTH, height=HEIGHT, format=rs.format.z16, framerate=FPS)
    cfg.enable_stream(rs.stream.infrared, 1, width=WIDTH, height=HEIGHT, format=rs.format.y8, framerate=FPS)
    cfg.enable_stream(rs.stream.infrared, 2, width=WIDTH, height=HEIGHT, format=rs.format.y8, framerate=FPS)
    cfg.enable_record_to_file(filename)
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Wide-baseline stereo implementation")
    parser.add_argument("-c", "--calibration", help=".npy file containing the numpy-serialized calibration data")
    args = parser.parse_args()
    main(args)
