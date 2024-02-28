import argparse
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


def main(args):
    ctx = rs.context()
    device_manager = DeviceManager(ctx)
    if device_manager.device_count(ctx) != 2:
        raise Exception(f"Unexpected number of devices (expected 2): {device_manager.device_count(ctx)}")
    try:
        left_serial = os.environ.get("RS_LEFT_SERIAL")
        right_serial = os.environ.get("RS_RIGHT_SERIAL")
        if left_serial and right_serial is not None:
            print(f"'RS_LEFT_SERIAL' and 'RS_RIGHT_SERIAL' environment variables are set:\n"
                  f"Left Device: {left_serial}\nRight Device: {right_serial}")
        else:
            left_serial, right_serial = DeviceManager.serial_selection()
    except Exception as e:
        print("Serial selection failed: \n", e)
        return

    device_pair = device_manager.create_device_pair(left_serial, right_serial)
    set_device_options(device_pair)

    if args.calibration:
        calibration_result = load_calibration_from_file(args.calibration)
        rectification_result = stereo_rectify(device_pair, calibration_result.image_size, calibration_result)
    else:
        calibration_result, rectification_result = run_camera_calibration(device_pair)

    cv.namedWindow(WINDOW_DEPTH_LEFT)
    cv.namedWindow(WINDOW_DEPTH_RIGHT)

    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    parent_dir = f"RECORDING_{timestamp}"
    os.mkdir(parent_dir)

    device_pair.start(WIDTH, HEIGHT, FPS, streams=(rs.stream.infrared, rs.stream.depth), record_to_directory=parent_dir)
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

    cv.destroyAllWindows()
    write_calibration_to_file(calibration_result, os.path.join(parent_dir, "Calibration"))
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
