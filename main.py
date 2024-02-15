import argparse
import datetime
import os
from dataclasses import dataclass
from typing import Tuple

import time

import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from scipy.interpolate import interp1d

from device_utility.DeviceManager import DeviceManager, DevicePair
from device_utility.camera_calibration import run_camera_calibration, stereo_rectify, RectificationResult, \
    load_calibration_from_file, CalibrationResult
from device_utility.utils import set_sensor_option, get_sensor_option, get_stereo_extrinsic

# GLOBALS
WINDOW_IR_L = "infrared left"
WINDOW_IR_R = "infrared right"
WINDOW_DEPTH = "depth"
WINDOW_NATIVE_LEFT = "native depth left"
WINDOW_NATIVE_RIGHT = "native depth right"
WINDOW_CONTROLS = "controls"

KEY_ESCAPE = 256
KEY_SPACE = 32

MOUSE_X, MOUSE_Y = 0, 0
MOUSE_OVER_WINDOW = ""

run = True

# assuming max depth of 13m here, needs adjustment depending on scene, maybe make it dynamic
MAP_DEPTH_M_TO_BYTE = interp1d([0, 13], [0, 255], bounds_error=False, fill_value=(0, 255))
MAP_DEPTH_MM_TO_BYTE = interp1d([0, 13000], [0, 255], bounds_error=False, fill_value=(0, 255))

# TODO save and load values
stereo_sgm = cv.StereoSGBM.create(
    minDisparity=6,
    numDisparities=16 * 6,
    blockSize=3,
    P1=8 * 3 * 3 ** 2,
    P2=31 * 3 * 3 ** 2,
    disp12MaxDiff=1,
    preFilterCap=1,
    uniquenessRatio=4,
    speckleWindowSize=100,
    speckleRange=1,
    mode=cv.STEREO_SGBM_MODE_SGBM
)

# basic block matcher
stereo_bm = cv.StereoBM.create(
    numDisparities=16 * 10,
    blockSize=7
)

stereo_algorithm = stereo_sgm


# based on https://learnopencv.com/depth-perception-using-stereo-camera-python-c/

def change_blockSize(value):
    odd_value = value if value % 2 == 1 else value + 1  # ensure block size odd
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


@dataclass()
class CameraParameters:
    left_intrinsics: rs.intrinsics
    right_intrinsics: rs.intrinsics
    left_pinhole_intrinsics: o3d.camera.PinholeCameraIntrinsic
    right_pinhole_intrinsics: o3d.camera.PinholeCameraIntrinsic
    left_stereo_extrinsic: rs.extrinsics
    right_stereo_extrinsic: rs.extrinsics
    image_size: Tuple[int, int]


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

    cv.line(left_rectified, (0, MOUSE_Y), (left_rectified.shape[1], MOUSE_Y), color=(0,), lineType=cv.LINE_4,
            thickness=1)
    cv.line(right_rectified, (0, MOUSE_Y), (left_rectified.shape[1], MOUSE_Y), color=(0,), lineType=cv.LINE_4,
            thickness=1)
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

    # set ir emitter to full power
    set_sensor_option(depth_sensor_left, rs.option.laser_power, 360)
    set_sensor_option(depth_sensor_right, rs.option.laser_power, 360)

    # turn off auto exposure
    set_sensor_option(depth_sensor_left, rs.option.enable_auto_exposure, 0)
    set_sensor_option(depth_sensor_right, rs.option.enable_auto_exposure, 0)


# pass window name as user data
def on_mouse(event, x, y, flags, user_data):
    global MOUSE_X, MOUSE_Y, MOUSE_OVER_WINDOW
    if event == cv.EVENT_MOUSEMOVE:
        MOUSE_X, MOUSE_Y = x, y
        MOUSE_OVER_WINDOW = user_data


def event_stop(vis, action, mod):
    global run
    run = False
    print("stop flag set")
    return False


def reset_view(vis: o3d.visualization.Visualizer, action, mod):
    # arg0 ist reset_bounding_box: bool
    vis.reset_view_point(True)

    # https://github.com/isl-org/Open3D/issues/1483#issuecomment-1423493280
    # set view point to roughly origin, facing negative Z -> the direction of the point cloud
    # this does not work because of a bug in open3d https://github.com/isl-org/Open3D/issues/1164
    # view_control: o3d.visualization.ViewControl = vis.get_view_control()
    # camera_parameters: o3d.camera.PinholeCameraParameters = view_control.convert_to_pinhole_camera_parameters()
    # camera_parameters.extrinsic = np.matrix("1 0 0 0; 0 -1 0 4; 0 0 -1 4; 0 0 0 1")
    # view_control.convert_from_pinhole_camera_parameters(camera_parameters)
    return True


def setup_gui(device_pair):
    cv.namedWindow(WINDOW_IR_L)
    cv.namedWindow(WINDOW_IR_R)
    cv.namedWindow(WINDOW_DEPTH)
    cv.namedWindow(WINDOW_NATIVE_LEFT)
    cv.namedWindow(WINDOW_NATIVE_RIGHT)
    cv.namedWindow(WINDOW_CONTROLS)
    cv.resizeWindow(WINDOW_CONTROLS, 600, 400)

    cv.setMouseCallback(WINDOW_DEPTH, on_mouse, WINDOW_DEPTH)
    cv.setMouseCallback(WINDOW_IR_L, on_mouse, WINDOW_IR_L)
    cv.setMouseCallback(WINDOW_IR_R, on_mouse, WINDOW_IR_R)
    cv.createTrackbar("blockSize", WINDOW_CONTROLS, stereo_algorithm.getBlockSize(), 15, change_blockSize)
    cv.setTrackbarMin("blockSize", WINDOW_CONTROLS, 5 if isinstance(stereo_algorithm, cv.StereoBM) else 3)
    if isinstance(stereo_algorithm, cv.StereoSGBM):
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

    # initialize open3d visualization
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_action_callback(KEY_ESCAPE, event_stop)
    vis.register_key_action_callback(KEY_SPACE, reset_view)
    vis.create_window("multicam point cloud", width=1280, height=720)
    render_option = vis.get_render_option()
    render_option.point_size = 2

    return vis


def compute_device_offset_transform(camera_params: CameraParameters, calib: CalibrationResult):
    """
    Return the 4x4 transformation matrix R_13=(R|t) in homogenous coordinates
    :param camera_params:
    :param calib: Calibration result from calibrating the inner cameras of the device pair
    :return:
    """
    R_12 = np.eye(4, dtype=np.float32)
    R_23 = np.eye(4, dtype=np.float32)

    # rs.extrinsics.rotation is column-major 3x3 matrix -> transpose to row major for compatibility with openCV
    R_12[:3, :3] = np.asarray(camera_params.left_stereo_extrinsic.rotation).reshape(3, 3).T
    R_12[:3, 3:4] = np.asarray(camera_params.left_stereo_extrinsic.translation).reshape(3, 1)

    # calib.R is already row-major as it was created by openCV
    R_23[:3, :3] = calib.R
    R_23[:3, 3:4] = calib.T

    # @ is shorthand for np.matmul(a, b)
    R_13 = R_23 @ R_12
    return R_13


def depth_to_point_cloud(depth: np.ndarray, intrinsic, extrinsic,
                         colormap: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    if colormap is not None:
        depth_image = o3d.geometry.Image(depth)
        # OpenCV is BGR, Open3D expects RGB
        depth_colormap_image = o3d.geometry.Image(colormap)
        depth_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=depth_colormap_image,
                                                                              depth=depth_image,
                                                                              depth_trunc=20,
                                                                              depth_scale=1000,
                                                                              convert_rgb_to_intensity=False)
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(depth_rgbd_image, intrinsic, extrinsic)
    else:
        depth_image = o3d.geometry.Image(depth)
        pc = o3d.geometry.PointCloud.create_from_depth_image(depth_image,
                                                             intrinsic,
                                                             extrinsic,
                                                             depth_scale=1000,
                                                             depth_trunc=20)
    return pc


def copy_to_point_cloud(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud):
    target.colors = source.colors
    target.points = source.points
    target.covariances = source.covariances
    target.normals = source.normals


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

    # we only need ir streams -> even only the outer streams
    # wide_stereo_points necessary to get intrinsics, although those could be gotten from ir streams as well
    device_pair.start(1280, 720, 15, streams=(rs.stream.infrared, rs.stream.depth))

    camera_parameters = get_camera_parameters(device_pair)

    wide_stereo_baseline = calibration_result.T[0, 0] if calibration_result.R_14 is None else calibration_result.R_14[
        0, 3]
    print(f"Left camera intrinsic parameters: {camera_parameters.left_intrinsics}")
    print(f"wide baseline: {wide_stereo_baseline:.4} m")

    device_offset_transform = compute_device_offset_transform(camera_parameters, calibration_result)

    vis = setup_gui(device_pair)

    # identity matrix with Y and Z flipped, left is origin
    # multiply right with left transform to flip Y [and Z] don't flip z, to be compatible with wide stereo
    point_cloud_transform_left = np.matrix("1 0 0 0;0 1 0 0; 0 0 1 0; 0 0 0 1")
    point_cloud_transform_right = device_offset_transform @ point_cloud_transform_left

    # we don't need the colormap in this step
    # create initial point clouds to be added to the scene
    left_frame, right_frame = device_pair.wait_for_frames()
    left_point_cloud = depth_to_point_cloud(np.asanyarray(left_frame.get_depth_frame().get_data()),
                                            camera_parameters.left_pinhole_intrinsics,
                                            point_cloud_transform_left)
    right_point_cloud = depth_to_point_cloud(np.asanyarray(right_frame.get_depth_frame().get_data()),
                                             camera_parameters.right_pinhole_intrinsics,
                                             point_cloud_transform_right)

    wide_stereo_points = wide_stereo_from_frames(left_frame,
                                                 right_frame,
                                                 wide_stereo_baseline,
                                                 camera_parameters.left_intrinsics.fx,
                                                 rectification_result)

    wide_point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(wide_stereo_points.reshape(-1, 3)))

    combined_point_cloud = o3d.geometry.PointCloud()

    # vis.add_geometry(left_point_cloud, True)
    # vis.add_geometry(right_point_cloud, True)
    # vis.add_geometry(wide_point_cloud, True)
    vis.add_geometry(combined_point_cloud, False)

    t0 = time.perf_counter_ns()
    global run
    while run:
        # TODO Screenshots, all frames
        left_frame, right_frame = device_pair.wait_for_frames()

        wide_stereo_points = wide_stereo_from_frames(left_frame,
                                                     right_frame,
                                                     wide_stereo_baseline,
                                                     camera_parameters.left_intrinsics.fx,
                                                     rectification_result)
        wide_stereo_points_threshold = np.where(wide_stereo_points[:, :, 2:3] > 2.1, wide_stereo_points, [0, 0, 0])

        # only map z-component of wide_stereo_points map
        depth_colormapped = cv.applyColorMap(MAP_DEPTH_M_TO_BYTE(wide_stereo_points_threshold).astype(np.uint8)[:, :, 2:3],
                                             cv.COLORMAP_JET)
        new_wide_point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(wide_stereo_points_threshold.reshape(-1, 3)))
        # try if this works, should work if order of points is kept intact
        new_wide_point_cloud.colors = o3d.utility.Vector3dVector(depth_colormapped.reshape(-1, 3))

        # copy_to_point_cloud(new_wide_point_cloud, wide_point_cloud)
        # vis.update_geometry(wide_point_cloud)

        depth_at_cursor = wide_stereo_points[np.clip(MOUSE_Y, 0, 719), np.clip(MOUSE_X, 0, 1279), 2]
        distance_at_cursor = np.linalg.norm(wide_stereo_points[np.clip(MOUSE_Y, 0, 719), np.clip(MOUSE_X, 0, 1279)])

        cv.putText(depth_colormapped, f"Depth/Distance: {depth_at_cursor:.3} / {distance_at_cursor:.3} m", (10, 40),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=2, color=(220, 220, 220), thickness=2)
        t1 = time.perf_counter_ns()
        td = (t1 - t0) / 1000000
        t0 = t1
        cv.putText(depth_colormapped, f"{td:.6} ms, {1000 / td:.3} FPS", (10, 80), fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=2, color=(220, 220, 220), thickness=2)
        if MOUSE_OVER_WINDOW != WINDOW_DEPTH:
            cv.drawMarker(depth_colormapped, [MOUSE_X, MOUSE_Y], [0, 0, 0], cv.MARKER_CROSS, markerSize=11, thickness=1)
        cv.imshow(WINDOW_DEPTH, depth_colormapped)

        # TODO color mapping, show wide_stereo_points streams, decimate native point clouds, integrate wide point cloud
        left_depth = np.asanyarray(left_frame.get_depth_frame().get_data())
        right_depth = np.asanyarray(right_frame.get_depth_frame().get_data())
        left_depth_threshold = np.where(left_depth < 2200, left_depth, 0)
        # right_depth_threshold = np.where(right_depth < 2200, right_depth, 0)

        left_depth_colormap = cv.applyColorMap(MAP_DEPTH_MM_TO_BYTE(left_depth_threshold).astype(np.uint8), cv.COLORMAP_JET)
        # right_depth_colormap = cv.applyColorMap(MAP_DEPTH_MM_TO_BYTE(right_depth_threshold).astype(np.uint8), cv.COLORMAP_JET)

        # show wide_stereo_points as BGR before converting to RGB, since imshow() expects BGR
        cv.imshow(WINDOW_NATIVE_LEFT, left_depth_colormap)
        # cv.imshow(WINDOW_NATIVE_RIGHT, right_depth_colormap)

        cv.cvtColor(left_depth_colormap, cv.COLOR_BGR2RGB, dst=left_depth_colormap)
        # cv.cvtColor(right_depth_colormap, cv.COLOR_BGR2RGB, dst=right_depth_colormap)

        # point cloud visuals
        new_left_point_cloud = depth_to_point_cloud(left_depth_threshold,
                                                    camera_parameters.left_pinhole_intrinsics,
                                                    point_cloud_transform_left,
                                                    colormap=left_depth_colormap)
        # new_right_point_cloud = depth_to_point_cloud(right_depth_threshold,
        #                                              camera_parameters.right_pinhole_intrinsics,
        #                                              point_cloud_transform_right,
        #                                              colormap=right_depth_colormap)
        # copy_to_point_cloud(new_left_point_cloud, left_point_cloud)
        # copy_to_point_cloud(new_right_point_cloud, right_point_cloud)

        # vis.update_geometry(left_point_cloud)
        # vis.update_geometry(right_point_cloud)

        # combine into one single way too big point cloud (naive)
        all_points = new_left_point_cloud.points
        # all_points.extend(new_right_point_cloud.points)
        all_points.extend(new_wide_point_cloud.points)

        all_colors = new_left_point_cloud.colors
        # all_colors.extend(new_right_point_cloud.colors)
        all_colors.extend(new_wide_point_cloud.colors)

        combined_point_cloud.points = all_points
        combined_point_cloud.colors = all_colors
        vis.update_geometry(combined_point_cloud)

        key = cv.pollKey()
        if key != -1:
            print(f"key code pressed: {key}")
        if key == 27 or not vis.poll_events():  # ESCAPE
            run = False
        if key == 115:  # s
            filename = f"Screenshot_{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S-%f')}.png"
            # add marker at mouse position to the saved image
            cv.drawMarker(depth_colormapped, [MOUSE_X, MOUSE_Y], [0, 0, 0], cv.MARKER_SQUARE, markerSize=6, thickness=1)
            cv.imwrite(filename, depth_colormapped)
            print(f"Screenshot saved as {filename}")
        if key == 112:  # p
            filename = f"PointCloud_{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S-%f')}.ply"
            o3d.io.write_point_cloud(filename, combined_point_cloud)
            print(f"Point-Cloud saved as {filename}")

        vis.update_renderer()

    vis.close()
    vis.destroy_window()
    cv.destroyAllWindows()
    print("Waiting for device pipelines to close...")
    device_pair.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Wide-baseline stereo implementation")
    parser.add_argument("-c", "--calibration", help=".npy file containing the numpy-serialized calibration data")
    args = parser.parse_args()
    main(args)
