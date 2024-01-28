import cv2 as cv
import numpy as np
import pyrealsense2 as rs
from device_utility.DeviceManager import DeviceManager, DevicePair
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

# GLOBALS
WINDOW_IR_L = "infrared left"
WINDOW_IR_R = "infrared right"
WINDOW_DEPTH = "depth"

MOUSE_X, MOUSE_Y = 0, 0

stereo_algorithm = cv.StereoSGBM.create(
    minDisparity=16,
    numDisparities=96,
    blockSize=11,
    P1=8 * 3 * 3 ** 2,
    P2=31 * 3 * 3 ** 2,
    disp12MaxDiff=0,
    preFilterCap=0,
    uniquenessRatio=0,
    speckleWindowSize=64,
    speckleRange=1,
    mode=cv.STEREO_SGBM_MODE_SGBM
)


def calculate_wide_stereo_depth(left: np.ndarray, right: np.ndarray, baseline, focal_length):
    global stereo_algorithm
    # omit rectification for now
    disp = stereo_algorithm.compute(left, right).astype(np.float32) / 16.0

    # TODO disp to depth using formula d=fb/z <=> z=fb/d
    fb = focal_length * abs(baseline)
    depth = fb * 1 / disp
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

def set_device_options(device_pair: DevicePair):
    depth_sensor_left: rs.depth_sensor = device_pair.left.pipeline_profile.get_device().first_depth_sensor()
    depth_sensor_right: rs.depth_sensor = device_pair.right.pipeline_profile.get_device().first_depth_sensor()
    if depth_sensor_left.supports(rs.option.emitter_enabled):
        depth_sensor_left.set_option(rs.option.emitter_enabled, True)
        depth_sensor_right.set_option(rs.option.emitter_enabled, True)
    if depth_sensor_left.supports(rs.option.emitter_always_on):
        depth_sensor_left.set_option(rs.option.emitter_always_on, True)
        depth_sensor_right.set_option(rs.option.emitter_always_on, True)

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

    device_pair = device_manager.enable_device_pair(left_serial, right_serial)

    set_device_options(device_pair)

    left_intrinsic: rs.intrinsics = device_pair.left.pipeline_profile.get_stream(
        rs.stream.depth).as_video_stream_profile().get_intrinsics()
    right_intrinsic: rs.intrinsics = device_pair.right.pipeline_profile.get_stream(
        rs.stream.depth).as_video_stream_profile().get_intrinsics()

    left_stereo_extrinsic = get_stereo_extrinsic(device_pair.left.pipeline_profile)
    right_stereo_extrinsic = get_stereo_extrinsic(device_pair.right.pipeline_profile)

    left_baseline = left_stereo_extrinsic.translation[0]

    wide_stereo_baseline = 3 * left_baseline

    cv.namedWindow(WINDOW_IR_L)
    cv.namedWindow(WINDOW_IR_R)
    cv.namedWindow(WINDOW_DEPTH)
    cv.setMouseCallback(WINDOW_DEPTH, on_mouse)

    map_range = interp1d([0, 10], [0, 255])
    map_depth_to_uint8 = lambda d: map_range(d).astype(np.uint8)

    stream_width = device_pair.left.pipeline_profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile().width()
    stream_height = device_pair.left.pipeline_profile.get_stream(rs.stream.infrared,
                                                                 1).as_video_stream_profile().height()
    stream_center = [stream_width // 2, stream_height // 2]

    run = True
    while run:
        left_frame, right_frame = device_pair.wait_for_frames()
        cv.imshow(WINDOW_IR_L, np.asanyarray(left_frame.get_infrared_frame(1).get_data()))
        cv.imshow(WINDOW_IR_R, np.asanyarray(right_frame.get_infrared_frame(2).get_data()))
        depth = wide_stereo_from_frames(left_frame, right_frame, wide_stereo_baseline, left_intrinsic.fx)
        depth_colormapped = cv.applyColorMap(map_depth_to_uint8(depth), cv.COLORMAP_JET)
        depth_at_cursor = depth[MOUSE_Y, MOUSE_X]
        cv.drawMarker(depth_colormapped, stream_center, [0, 0, 0], cv.MARKER_SQUARE, markerSize=8, thickness=2)
        cv.putText(depth_colormapped, f"{depth_at_cursor:.3} m", [40, 40], fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1.1, color=[0, 0, 0], thickness=2)
        cv.imshow(WINDOW_DEPTH, depth_colormapped)

        if cv.waitKey(1) == 27:
            run = False

    cv.destroyAllWindows()
    print("Waiting for device pipelines to close...")
    device_pair.stop()


if __name__ == '__main__':
    main()
