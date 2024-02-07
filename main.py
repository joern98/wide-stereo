import argparse
import datetime
import os
import time

import cv2 as cv
import numpy as np
import pyrealsense2 as rs
from scipy.interpolate import interp1d

from device_utility.DeviceManager import DeviceManager, DevicePair
from device_utility.camera_calibration import run_camera_calibration, stereo_rectify, RectificationResult, \
    load_calibration_from_file
from device_utility.utils import set_sensor_option, get_sensor_option, get_stereo_extrinsic

# GLOBALS
WINDOW_IR_L = "infrared left"
WINDOW_IR_R = "infrared right"
WINDOW_DEPTH = "depth"
WINDOW_CONTROLS = "controls"

MOUSE_X, MOUSE_Y = 0, 0
MOUSE_OVER_WINDOW = ""

# TODO test StereoBM just for completeness
stereo_algorithm = cv.StereoSGBM.create(
    minDisparity=1,
    numDisparities=16 * 10,
    blockSize=5,
    P1=8 * 3 * 3 ** 2,
    P2=31 * 3 * 3 ** 2,
    disp12MaxDiff=-1,
    preFilterCap=0,
    uniquenessRatio=8,
    speckleWindowSize=100,
    speckleRange=1,
    mode=cv.STEREO_SGBM_MODE_SGBM
)


# based on https://learnopencv.com/depth-perception-using-stereo-camera-python-c/

def change_blockSize(value):
    # odd_value = value if value % 2 == 1 else value+1  # ensure block size odd
    odd_value = value
    stereo_algorithm.setBlockSize(odd_value)
    cv.setTrackbarPos("blockSize", WINDOW_CONTROLS, odd_value)


def change_P1(value):
    p2 = stereo_algorithm.getP2()
    stereo_algorithm.setP1(min(p2 - 1, value))


def change_P2(value):
    p1 = stereo_algorithm.getP1()
    stereo_algorithm.setP2(max(value, p1 + 1))  # ensure P1 < P2


def change_disp12MaxDiff(value):
    stereo_algorithm.setDisp12MaxDiff(value)


def change_preFilterCap(value):
    stereo_algorithm.setPreFilterCap(value)


def change_uniquenessRatio(value):
    stereo_algorithm.setUniquenessRatio(value)


def change_speckleWindowSize(value):
    stereo_algorithm.setSpeckleWindowSize(value)


def change_speckleRange(value):
    stereo_algorithm.setSpeckleRange(value)


# device_pair passed as userdata
def change_exposure_time(value, device_pair: DevicePair):
    depth_sensor_left: rs.depth_sensor = device_pair.left.device.first_depth_sensor()
    depth_sensor_right: rs.depth_sensor = device_pair.right.device.first_depth_sensor()
    set_sensor_option(depth_sensor_left, rs.option.exposure, value)
    set_sensor_option(depth_sensor_right, rs.option.exposure, value)


def calculate_wide_stereo_depth(left: np.ndarray, right: np.ndarray, rectification: RectificationResult):
    global stereo_algorithm
    # omit rectification for now
    disp = stereo_algorithm.compute(left, right).astype(np.float32) / 16.0

    # disp to depth using formula d=fb/z <=> z=fb/d
    # Q: Output 4×4 disparity-to-depth mapping matrix (see reprojectImageTo3D).
    # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga1bc1152bd57d63bc524204f21fde6e02

    depth = cv.reprojectImageTo3D(disp, rectification.Q)
    return depth


def wide_stereo_from_frames(left: rs.composite_frame, right: rs.composite_frame, baseline, focal_length,
                            rectification: RectificationResult):
    ir_frame_left: rs.video_frame = left.get_infrared_frame(1)  # left most IR stream
    ir_frame_right: rs.video_frame = right.get_infrared_frame(2)  # right most IR stream
    left_array = np.asanyarray(ir_frame_left.get_data())
    right_array = np.asanyarray(ir_frame_right.get_data())

    # apply stereo rectification
    # Reference: https://learnopencv.com/depth-perception-using-stereo-camera-python-c/
    # 	cv.remap(	src, map1, map2, interpolation[, dst[, borderMode[, borderValue]]]	) -> 	dst
    # https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
    left_rectified = cv.remap(left_array,
                              rectification.left_map_x,
                              rectification.left_map_y,
                              interpolation=cv.INTER_LANCZOS4,
                              borderMode=cv.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))

    right_rectified = cv.remap(right_array,
                               rectification.right_map_x,
                               rectification.right_map_y,
                               interpolation=cv.INTER_LANCZOS4,
                               borderMode=cv.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))

    wide_stereo_result = calculate_wide_stereo_depth(left_rectified, right_rectified, rectification)

    cv.line(left_rectified, (0, MOUSE_Y), (left_rectified.shape[1], MOUSE_Y), color=(0,), lineType=cv.LINE_4, thickness=1)
    cv.line(right_rectified, (0, MOUSE_Y), (left_rectified.shape[1], MOUSE_Y), color=(0,), lineType=cv.LINE_4, thickness=1)
    if MOUSE_OVER_WINDOW != WINDOW_IR_L:
        # images are greyscale, so no colored line possible
        cv.drawMarker(left_rectified, (MOUSE_X, MOUSE_Y), (0,), cv.MARKER_CROSS, markerSize=11, thickness=1)

    if MOUSE_OVER_WINDOW != WINDOW_IR_R:
        cv.drawMarker(right_rectified, (MOUSE_X, MOUSE_Y), (0,), cv.MARKER_CROSS, markerSize=11, thickness=1)



    cv.imshow(WINDOW_IR_L, left_rectified)
    cv.imshow(WINDOW_IR_R, right_rectified)

    return wide_stereo_result


# TODO turn of auto exposure and set both the same ->
#  better would be just having the left camera control exposure time and writing left params to right camera
def set_device_options(device_pair: DevicePair):
    depth_sensor_left: rs.depth_sensor = device_pair.left.device.first_depth_sensor()
    depth_sensor_right: rs.depth_sensor = device_pair.right.device.first_depth_sensor()
    left_supported = depth_sensor_left.get_supported_options()
    if depth_sensor_left.supports(rs.option.emitter_enabled):
        depth_sensor_left.set_option(rs.option.emitter_enabled, True)
        depth_sensor_right.set_option(rs.option.emitter_enabled, True)
    if depth_sensor_left.supports(rs.option.emitter_always_on):
        depth_sensor_left.set_option(rs.option.emitter_always_on, True)
        depth_sensor_right.set_option(rs.option.emitter_always_on, True)

    # turn off auto exposure
    set_sensor_option(depth_sensor_left, rs.option.enable_auto_exposure, 0)
    set_sensor_option(depth_sensor_right, rs.option.enable_auto_exposure, 0)

    gain_l = get_sensor_option(depth_sensor_left, rs.option.gain)
    gain_r = get_sensor_option(depth_sensor_right, rs.option.gain)
    print(gain_l)
    print(gain_r)

# pass window name as user data
def on_mouse(event, x, y, flags, user_data):
    global MOUSE_X, MOUSE_Y, MOUSE_OVER_WINDOW
    if event == cv.EVENT_MOUSEMOVE:
        MOUSE_X, MOUSE_Y = x, y
        MOUSE_OVER_WINDOW = user_data


def main(args):
    ctx = rs.context()
    device_manager = DeviceManager(ctx)
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

    # we only need ir streams -> even only the outer streams
    # depth necessary to get intrinsics, although those could be gotten from ir streams as well
    device_pair.start(1280, 720, 15, streams=(rs.stream.infrared, rs.stream.depth))

    left_intrinsic: rs.intrinsics = device_pair.left.pipeline_profile.get_stream(
        rs.stream.depth).as_video_stream_profile().get_intrinsics()
    right_intrinsic: rs.intrinsics = device_pair.right.pipeline_profile.get_stream(
        rs.stream.depth).as_video_stream_profile().get_intrinsics()

    left_stereo_extrinsic = get_stereo_extrinsic(device_pair.left.pipeline_profile)
    right_stereo_extrinsic = get_stereo_extrinsic(device_pair.right.pipeline_profile)

    wide_stereo_baseline = calibration_result.T[0, 0] if calibration_result.R_14 is None else calibration_result.R_14[0, 3]
    print(f"Left camera intrinsic parameters: {left_intrinsic}")
    print(f"wide baseline: {wide_stereo_baseline:.4} m")

    cv.namedWindow(WINDOW_IR_L)
    cv.namedWindow(WINDOW_IR_R)
    cv.namedWindow(WINDOW_DEPTH)
    cv.namedWindow(WINDOW_CONTROLS)
    cv.resizeWindow(WINDOW_CONTROLS, 600, 400)

    cv.setMouseCallback(WINDOW_DEPTH, on_mouse, WINDOW_DEPTH)
    cv.setMouseCallback(WINDOW_IR_L, on_mouse, WINDOW_IR_L)
    cv.setMouseCallback(WINDOW_IR_R, on_mouse, WINDOW_IR_R)

    cv.createTrackbar("blockSize", WINDOW_CONTROLS, stereo_algorithm.getBlockSize(), 15, change_blockSize)
    cv.createTrackbar("p1", WINDOW_CONTROLS, stereo_algorithm.getP1(), 1000, change_P1)
    cv.createTrackbar("p2", WINDOW_CONTROLS, stereo_algorithm.getP2(), 3000, change_P2)
    cv.createTrackbar("disp12MaxDiff", WINDOW_CONTROLS, stereo_algorithm.getDisp12MaxDiff(), 16, change_disp12MaxDiff)
    cv.createTrackbar("preFilterCap", WINDOW_CONTROLS, stereo_algorithm.getPreFilterCap(), 16, change_preFilterCap)
    cv.createTrackbar("uniquenessRatio", WINDOW_CONTROLS, stereo_algorithm.getUniquenessRatio(), 16,
                      change_uniquenessRatio)
    cv.createTrackbar("speckleWindowSize", WINDOW_CONTROLS, stereo_algorithm.getSpeckleWindowSize(), 200,
                      change_speckleWindowSize)
    cv.createTrackbar("speckleRange", WINDOW_CONTROLS, stereo_algorithm.getSpeckleRange(), 3, change_speckleRange)

    # exposure unit is microseconds -> [0, 166000] 166ms
    cv.createTrackbar("exposure", WINDOW_CONTROLS, 0, 166000, lambda v: change_exposure_time(v, device_pair))
    cv.setTrackbarPos("exposure", WINDOW_CONTROLS,
                      int(get_sensor_option(device_pair.left.device.first_depth_sensor(), rs.option.exposure)))

    # assuming max depth of 12m here, needs adjustment depending on scene, maybe make it dynamic
    map_range = interp1d([0, 12], [0, 255], bounds_error=False, fill_value=(0, 255))
    map_depth_to_uint8 = lambda d: map_range(d).astype(np.uint8)

    stream_width = device_pair.left.pipeline_profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile().width()
    stream_height = device_pair.left.pipeline_profile.get_stream(rs.stream.infrared,
                                                                 1).as_video_stream_profile().height()
    stream_center = [stream_width // 2, stream_height // 2]

    run = True
    t0 = time.perf_counter_ns()
    while run:
        # TODO Screenshots, all frames
        left_frame, right_frame = device_pair.wait_for_frames()
        # cv.imshow(WINDOW_IR_L, np.asanyarray(left_frame.get_infrared_frame(1).get_data()))
        # cv.imshow(WINDOW_IR_R, np.asanyarray(right_frame.get_infrared_frame(2).get_data()))

        depth = wide_stereo_from_frames(left_frame,
                                        right_frame,
                                        wide_stereo_baseline,
                                        left_intrinsic.fx,
                                        rectification_result)

        # only map z-component of depth map
        depth_colormapped = cv.applyColorMap(map_depth_to_uint8(depth)[:, :, 2:3], cv.COLORMAP_JET)
        depth_at_cursor = depth[np.clip(MOUSE_Y, 0, 719), np.clip(MOUSE_X, 0, 1279), 2]

        cv.putText(depth_colormapped, f"{depth_at_cursor:.3} m", [10, 40], fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=[0, 0, 0], thickness=1)
        t1 = time.perf_counter_ns()
        td = (t1 - t0) / 1000000
        t0 = t1
        cv.putText(depth_colormapped, f"{td:.6} ms, {1000 / td:.3} FPS", [10, 80], fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=[0, 0, 0], thickness=1)
        if MOUSE_OVER_WINDOW != WINDOW_DEPTH:
            cv.drawMarker(depth_colormapped, [MOUSE_X, MOUSE_Y], [0, 0, 0], cv.MARKER_CROSS, markerSize=11, thickness=1)
        cv.imshow(WINDOW_DEPTH, depth_colormapped)

        key = cv.pollKey()
        if key != -1:
            print(f"key code pressed: {key}")
        if key == 27:  # ESCAPE
            run = False

        if key == 115:  # s
            filename = f"Screenshot_{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S-%f')}.png"
            # add marker at mouse position to the saved image
            cv.drawMarker(depth_colormapped, [MOUSE_X, MOUSE_Y], [0, 0, 0], cv.MARKER_SQUARE, markerSize=6, thickness=1)
            cv.imwrite(filename, depth_colormapped)
            print(f"Screenshot saved as {filename}")

    cv.destroyAllWindows()
    print("Waiting for device pipelines to close...")
    device_pair.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Wide-baseline stereo implementation")
    parser.add_argument("-c", "--calibration", help=".npy file containing the numpy-serialized calibration data")
    args = parser.parse_args()
    main(args)
