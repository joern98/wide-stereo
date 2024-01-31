import time
import datetime

import cv2 as cv
import numpy as np
import pyrealsense2 as rs
from device_utility.DeviceManager import DeviceManager, DevicePair
from device_utility.utils import set_sensor_option, get_sensor_option
from scipy.interpolate import interp1d

from camera_calibration import run_camera_calibration

# GLOBALS
WINDOW_IR_L = "infrared left"
WINDOW_IR_R = "infrared right"
WINDOW_DEPTH = "depth"
WINDOW_CONTROLS = "controls"

MOUSE_X, MOUSE_Y = 0, 0

# TODO test StereoBM just for completeness
stereo_algorithm = cv.StereoSGBM.create(
    minDisparity=1,
    numDisparities=16 * 6,
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


def calculate_wide_stereo_depth(left: np.ndarray, right: np.ndarray, baseline, focal_length):
    global stereo_algorithm
    # omit rectification for now
    disp = stereo_algorithm.compute(left, right).astype(np.float32) / 16.0

    # disp to depth using formula d=fb/z <=> z=fb/d
    fb = focal_length * abs(baseline)
    depth = fb / disp
    return depth


def wide_stereo_from_frames(left: rs.composite_frame, right: rs.composite_frame, baseline, focal_length):
    ir_frame_left: rs.video_frame = left.get_infrared_frame(1)  # left most IR stream
    ir_frame_right: rs.video_frame = right.get_infrared_frame(2)  # right most IR stream
    left_array = np.asanyarray(ir_frame_left.get_data())
    right_array = np.asanyarray(ir_frame_right.get_data())
    return calculate_wide_stereo_depth(left_array, right_array, baseline, focal_length)


def get_stereo_extrinsic(profile: rs.pipeline_profile) -> rs.extrinsics:
    # https://dev.intelrealsense.com/docs/api-how-to#get-disparity-baseline
    ir0_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    ir1_profile = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    e = ir0_profile.get_extrinsics_to(ir1_profile)
    return e


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


def on_mouse(event, x, y, flags, user_data):
    global MOUSE_X, MOUSE_Y
    if event == cv.EVENT_MOUSEMOVE:
        MOUSE_X, MOUSE_Y = x, y


def main():
    ctx = rs.context()
    device_manager = DeviceManager(ctx)
    try:
        left_serial, right_serial = DeviceManager.serial_selection()
    except Exception as e:
        print("Serial selection failed: \n", e)
        return

    device_pair = device_manager.create_device_pair(left_serial, right_serial)
    set_device_options(device_pair)

    run_camera_calibration(device_pair)

    device_pair.start(1280, 720, 15)

    left_intrinsic: rs.intrinsics = device_pair.left.pipeline_profile.get_stream(
        rs.stream.depth).as_video_stream_profile().get_intrinsics()
    right_intrinsic: rs.intrinsics = device_pair.right.pipeline_profile.get_stream(
        rs.stream.depth).as_video_stream_profile().get_intrinsics()

    left_stereo_extrinsic = get_stereo_extrinsic(device_pair.left.pipeline_profile)
    right_stereo_extrinsic = get_stereo_extrinsic(device_pair.right.pipeline_profile)

    left_baseline = left_stereo_extrinsic.translation[0]

    wide_stereo_baseline = 3 * left_baseline

    print(f"Left camera intrinsic parameters: {left_intrinsic}")
    print(f"Wide stereo baseline: {abs(wide_stereo_baseline)} m")

    cv.namedWindow(WINDOW_IR_L)
    cv.namedWindow(WINDOW_IR_R)
    cv.namedWindow(WINDOW_DEPTH)
    cv.namedWindow(WINDOW_CONTROLS)
    cv.resizeWindow(WINDOW_CONTROLS, 600, 400)

    cv.setMouseCallback(WINDOW_DEPTH, on_mouse)

    cv.createTrackbar("blockSize", WINDOW_CONTROLS, stereo_algorithm.getBlockSize(), 15, change_blockSize)
    cv.createTrackbar("p1", WINDOW_CONTROLS, stereo_algorithm.getP1(), 1000, change_P1)
    cv.createTrackbar("p2", WINDOW_CONTROLS, stereo_algorithm.getP2(), 1000, change_P2)
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
        cv.imshow(WINDOW_IR_L, np.asanyarray(left_frame.get_infrared_frame(1).get_data()))
        cv.imshow(WINDOW_IR_R, np.asanyarray(right_frame.get_infrared_frame(2).get_data()))
        depth = wide_stereo_from_frames(left_frame, right_frame, wide_stereo_baseline, left_intrinsic.fx)
        depth_colormapped = cv.applyColorMap(map_depth_to_uint8(depth), cv.COLORMAP_JET)
        depth_at_cursor = depth[np.clip(MOUSE_Y, 0, 719), np.clip(MOUSE_X, 0, 1279)]

        cv.putText(depth_colormapped, f"{depth_at_cursor:.3} m", [10, 40], fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=[0, 0, 0], thickness=1)
        t1 = time.perf_counter_ns()
        td = (t1 - t0) / 1000000
        t0 = t1
        cv.putText(depth_colormapped, f"{td:.6} ms, {1000 / td:.3} FPS", [10, 80], fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=[0, 0, 0], thickness=1)
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
    main()
