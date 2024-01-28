import cv2 as cv
import numpy as np
import pyrealsense2 as rs
from device_utility.DeviceManager import DeviceManager

# GLOBALS
WINDOW_NAME = "wide stereo"


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


def main():
    ctx = rs.context()
    device_manager = DeviceManager(ctx)
    try:
        left_serial, right_serial = DeviceManager.serial_selection()
    except Exception as e:
        print("Serial selection failed: \n", e)
        return

    device_pair = device_manager.enable_device_pair(left_serial, right_serial)

    left_intrinsic: rs.intrinsics = device_pair.left.pipeline_profile.get_stream(
        rs.stream.depth).as_video_stream_profile().get_intrinsics()
    right_intrinsic: rs.intrinsics = device_pair.right.pipeline_profile.get_stream(
        rs.stream.depth).as_video_stream_profile().get_intrinsics()

    left_stereo_extrinsic = get_stereo_extrinsic(device_pair.left.pipeline_profile)
    right_stereo_extrinsic = get_stereo_extrinsic(device_pair.right.pipeline_profile)

    left_baseline = left_stereo_extrinsic.translation[0]

    wide_stereo_baseline = 3 * left_baseline

    cv.namedWindow(WINDOW_NAME)

    run = True
    while run:
        left_frame, right_frame = device_pair.wait_for_frames()
        cv.imshow(WINDOW_NAME, np.asanyarray(left_frame.get_infrared_frame(1).get_data()))

        if cv.waitKey(1) == 27:
            run = False

    cv.destroyWindow(WINDOW_NAME)
    print("Waiting for device pipelines to close...")
    device_pair.stop()


if __name__ == '__main__':
    main()
