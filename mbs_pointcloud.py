import argparse
import json
import math
import os.path
import os.path as path
import time
from datetime import datetime

import cv2 as cv
import numpy as np
import open3d.geometry
from line_profiler_pycharm import profile
from scipy.interpolate import interp1d
import open3d as o3d

from device_utility.camera_calibration import load_calibration_from_file, stereo_rectify
from utility import save_point_cloud

WINDOW_LEFT_IR_1 = "left IR 1"
WINDOW_LEFT_IR_2 = "left IR 2"
WINDOW_RIGHT_IR_1 = "right IR 1"
WINDOW_RIGHT_IR_2 = "right IR 2"
WINDOW_DEPTH_LEFT = "left depth"
WINDOW_DEPTH_WIDE = "wide depth"

MAP_DEPTH_M_TO_BYTE = interp1d([0, 4], [0, 255], bounds_error=False, fill_value=(0, 255))

KEY_ESCAPE = 256
KEY_SPACE = 32


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


def depth_to_point_cloud(depth: np.ndarray, intrinsic, extrinsic,
                         colormap: np.ndarray | None = None, depth_scale=1) -> o3d.geometry.PointCloud:
    if colormap is not None:
        depth_image = o3d.geometry.Image(depth)
        # OpenCV is BGR, Open3D expects RGB
        depth_colormap_image = o3d.geometry.Image(colormap)
        depth_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=depth_colormap_image,
                                                                              depth=depth_image,
                                                                              depth_trunc=20,
                                                                              depth_scale=depth_scale,
                                                                              convert_rgb_to_intensity=False)
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(depth_rgbd_image, intrinsic, extrinsic)
    else:
        depth_image = o3d.geometry.Image(depth)
        pc = o3d.geometry.PointCloud.create_from_depth_image(depth_image,
                                                             intrinsic,
                                                             extrinsic,
                                                             depth_scale=depth_scale,
                                                             depth_trunc=20)
    return pc


def create_pinhole_intrinsic_from_dict(intrinsic_dict, image_size):
    return o3d.camera.PinholeCameraIntrinsic(width=image_size[0],
                                             height=image_size[1],
                                             cx=intrinsic_dict["ppx"],
                                             cy=intrinsic_dict["ppy"],
                                             fx=intrinsic_dict["fx"],
                                             fy=intrinsic_dict["fy"])


OUTPUT_DIRECTORY = None


def ensure_output_directory(root_directory):
    global OUTPUT_DIRECTORY
    if OUTPUT_DIRECTORY is None:
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        pathname = path.join(root_directory, f"Output_{timestamp}_mbs_pointcloud")
        os.mkdir(pathname)
        OUTPUT_DIRECTORY = pathname
    return OUTPUT_DIRECTORY


@profile
def double_stereo(ir_0, ir_1, ir_2, ir_3, calibration_result, camera_parameters,
                  z_near=0.5, z_far=10.0, error_threshold=0.02):
    # narrow stereo
    DISPARITY_ERROR = 0.5
    f = camera_parameters["left_intrinsics"]["fx"]
    b = abs(camera_parameters["left_stereo_extrinsics"]["t"][0])
    bf = b * f
    z_where_error_too_big = math.sqrt(bf * error_threshold / DISPARITY_ERROR)
    d_min = int(bf / z_where_error_too_big)  # minimum disparity
    d_max = int(bf / z_near)  # maximum disparity from where wide stereo is used
    num_disp = 16 * ((d_max - d_min) // 16 + 1)  # num disparities rounded up to the next multiple of 16
    print(
        f"Computing narrow stereo with d_min: {d_min}, d_max: {d_max} and num_disp: {num_disp} for threshold depth z: {z_where_error_too_big},"
        f"min z: {bf / d_max}, max z: {bf / d_min}")
    stereo95 = cv.StereoSGBM.create(
        minDisparity=d_min,
        numDisparities=num_disp,
        blockSize=3,
        P1=100,
        P2=400,
        disp12MaxDiff=4,
        preFilterCap=1,
        uniquenessRatio=5,
        speckleWindowSize=174,
        speckleRange=1,
        mode=cv.STEREO_SGBM_MODE_HH
    )

    # compute stereo left device
    disparity_narrow = stereo95.compute(ir_0, ir_1).astype(np.float32) / 16.0
    depth_narrow = bf / disparity_narrow

    # remove plane at z_max result from constraining disparity
    max_depth_value = np.max(depth_narrow)
    depth_narrow = np.where(depth_narrow == max_depth_value, 0, depth_narrow)

    pinhole_intrinsics = create_pinhole_intrinsic_from_dict(camera_parameters["left_intrinsics"],
                                                            camera_parameters["image_size"])
    narrow_point_cloud = depth_to_point_cloud(depth_narrow, pinhole_intrinsics, np.eye(4))

    # wide stereo
    rectification = stereo_rectify(image_size=camera_parameters["image_size"], calib=calibration_result)
    f = rectification.P_left[0, 0]  # focal length of the rectified images
    b = abs(calibration_result.R_14[0, 3])
    bf = b * f
    d_min = int(bf / z_far)
    d_max = int(bf / z_where_error_too_big)
    num_disp = 16 * ((d_max - d_min) // 16 + 1)
    print(
        f"Computing wide stereo with d_min: {d_min}, d_max: {d_max} and num_disp: {num_disp} for threshold depth z: {z_where_error_too_big},"
        f"min z: {bf / d_max}, max z: {bf / d_min}")
    stereo285 = cv.StereoSGBM.create(
        minDisparity=d_min,
        numDisparities=num_disp,
        blockSize=3,
        P1=100,
        P2=400,
        disp12MaxDiff=1,
        preFilterCap=1,
        uniquenessRatio=5,
        speckleWindowSize=128,
        speckleRange=1,
        mode=cv.STEREO_SGBM_MODE_HH

    )

    # compute stereo wide
    # rectify first
    left_rectified = cv.remap(ir_0,
                              rectification.left_map_x,
                              rectification.left_map_y,
                              interpolation=cv.INTER_LANCZOS4,
                              borderMode=cv.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
    right_rectified = cv.remap(ir_3,
                               rectification.right_map_x,
                               rectification.right_map_y,
                               interpolation=cv.INTER_LANCZOS4,
                               borderMode=cv.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))
    wide_disparity_raw = stereo285.compute(left_rectified, right_rectified)
    wide_disparity = wide_disparity_raw.astype(np.float32) / 16.0
    points285 = cv.reprojectImageTo3D(wide_disparity, rectification.Q, handleMissingValues=True)
    points285_flat = points285.reshape(-1, 3)
    # find indices where reprojectImageTo3D() has set Z to 10000 to mark invalid point
    invalid_indices = np.nonzero(points285_flat[:, 2:3] == 10000)
    points285_valid_only = np.delete(points285_flat, invalid_indices[0], 0)
    wide_point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points285_valid_only.reshape(-1, 3)))
    depth_wide = points285[:, :, 2:3]

    # combine point clouds
    combined_point_cloud = o3d.geometry.PointCloud(narrow_point_cloud)
    combined_point_cloud.points.extend(wide_point_cloud.points)

    return combined_point_cloud, narrow_point_cloud, wide_point_cloud, depth_narrow, depth_wide


def compute_isolated_point_clouds(ir_0, ir_1, ir_2, ir_3, native_depth, calibration_result, camera_parameters,
                                  z_near=0.5):
    f = camera_parameters["left_intrinsics"]["fx"]
    b = abs(camera_parameters["left_stereo_extrinsics"]["t"][0])
    bf = b * f
    d_max = int(bf / z_near)
    stereo95 = cv.StereoSGBM.create(
        minDisparity=0,
        numDisparities=(d_max // 16 + 1) * 16,
        blockSize=13,
        P1=50,
        P2=200,
        disp12MaxDiff=-1,
        preFilterCap=-1,
        uniquenessRatio=5,
        speckleWindowSize=174,
        speckleRange=1,
        mode=cv.STEREO_SGBM_MODE_HH
    )

    # compute stereo left device
    disparity_narrow = stereo95.compute(ir_0[:, :, 0], ir_1[:, :, 0]).astype(np.float32) / 16.0
    depth_narrow = bf / disparity_narrow

    # remove plane at z_max result from constraining disparity
    max_depth_value = np.max(depth_narrow)
    depth_narrow = np.where(depth_narrow == max_depth_value, 0, depth_narrow)

    pinhole_intrinsics = create_pinhole_intrinsic_from_dict(camera_parameters["left_intrinsics"],
                                                            camera_parameters["image_size"])
    narrow_point_cloud = depth_to_point_cloud(depth_narrow, pinhole_intrinsics, np.eye(4))

    rectification = stereo_rectify(image_size=camera_parameters["image_size"], calib=calibration_result)
    f = rectification.P_left[0, 0]  # focal length of the rectified images
    b = abs(calibration_result.R_14[0, 3])
    bf = b * f
    d_max = int(bf / z_near)
    stereo285 = cv.StereoSGBM.create(
        minDisparity=0,
        numDisparities=(d_max // 16 + 1) * 16,
        blockSize=3,
        P1=100,
        P2=400,
        disp12MaxDiff=4,
        preFilterCap=1,
        uniquenessRatio=5,
        speckleWindowSize=174,
        speckleRange=1,
        mode=cv.STEREO_SGBM_MODE_HH
    )

    # compute stereo wide
    # rectify first
    left_rectified = cv.remap(ir_0,
                              rectification.left_map_x,
                              rectification.left_map_y,
                              interpolation=cv.INTER_LANCZOS4,
                              borderMode=cv.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
    right_rectified = cv.remap(ir_3,
                               rectification.right_map_x,
                               rectification.right_map_y,
                               interpolation=cv.INTER_LANCZOS4,
                               borderMode=cv.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))
    wide_disparity_raw = stereo285.compute(left_rectified, right_rectified)
    wide_disparity = wide_disparity_raw.astype(np.float32) / 16.0
    points285 = cv.reprojectImageTo3D(wide_disparity, rectification.Q, handleMissingValues=True)
    points285_flat = points285.reshape(-1, 3)
    # find indices where reprojectImageTo3D() has set Z to 10000 to mark invalid point
    invalid_indices = np.nonzero(points285_flat[:, 2:3] == 10000)
    points285_valid_only = np.delete(points285_flat, invalid_indices[0], 0)
    wide_point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points285_valid_only.reshape(-1, 3)))
    wide_point_cloud.remove_non_finite_points()
    depth_wide = points285[:, :, 2:3]

    native_point_cloud = depth_to_point_cloud(native_depth, pinhole_intrinsics, np.eye(4), depth_scale=1000)

    return narrow_point_cloud, wide_point_cloud, depth_narrow, depth_wide, native_point_cloud


def main(args):
    calibration_result, camera_parameters, \
        left_ir_1, left_ir_2, right_ir_1, right_ir_2, \
        left_native_depth, right_native_depth = load_data(args.directory)

    # show raw infrared images
    cv.imshow(WINDOW_LEFT_IR_1, left_ir_1)
    cv.imshow(WINDOW_LEFT_IR_2, left_ir_2)
    cv.imshow(WINDOW_RIGHT_IR_1, right_ir_1)
    cv.imshow(WINDOW_RIGHT_IR_2, right_ir_2)

    combined_pointcloud, narrow_pointcloud, wide_pointcloud, narrow_depth, wide_depth = double_stereo(left_ir_1,
                                                                                                      left_ir_2,
                                                                                                      right_ir_1,
                                                                                                      right_ir_2,
                                                                                                      calibration_result,
                                                                                                      camera_parameters,
                                                                                                      z_near=0.5,
                                                                                                      z_far=16.0,
                                                                                                      error_threshold=0.03)


    full_narrow_point_cloud, full_wide_point_cloud, full_narrow_depth, full_wide_depth, native_point_cloud = \
        compute_isolated_point_clouds(left_ir_1, left_ir_2, right_ir_1, right_ir_2, left_native_depth,
                                      calibration_result, camera_parameters,
                                      z_near=0.5)

    depth_narrow_colorized = cv.applyColorMap(MAP_DEPTH_M_TO_BYTE(narrow_depth).astype(np.uint8), cv.COLORMAP_JET)
    cv.imshow(WINDOW_DEPTH_LEFT, depth_narrow_colorized)

    depth_wide_colorized = cv.applyColorMap(MAP_DEPTH_M_TO_BYTE(wide_depth).astype(np.uint8), cv.COLORMAP_JET)
    cv.imshow(WINDOW_DEPTH_WIDE, depth_wide_colorized)

    full_depth_narrow_colorized = cv.applyColorMap(MAP_DEPTH_M_TO_BYTE(full_narrow_depth).astype(np.uint8),
                                                   cv.COLORMAP_JET)
    full_depth_wide_colorized = cv.applyColorMap(MAP_DEPTH_M_TO_BYTE(full_wide_depth).astype(np.uint8), cv.COLORMAP_JET)

    def on_mouse(event, x, y, flags, user_data):
        if event == cv.EVENT_MOUSEMOVE:
            print(
                f"left depth: {narrow_depth[y, x]} m | wide depth: {wide_depth[y, x]} m | diff: {narrow_depth[y, x] - wide_depth[y, x]} | left native "
                f"depth: {left_native_depth[y, x] / 1000} m")

    run = True

    cv.setMouseCallback(WINDOW_DEPTH_LEFT, on_mouse)
    cv.setMouseCallback(WINDOW_DEPTH_WIDE, on_mouse)

    def vis_close(vis, action, mod):
        nonlocal run
        run = False
        vis.destroy_window()
        return False

    def output_dir():
        return ensure_output_directory(args.directory)

    while run:
        key = cv.waitKey(1)
        if key == 27:  # ESCAPE
            run = False
        if key == ord('s'):
            # save combined point clouds and parts
            save_point_cloud(narrow_pointcloud, "PointCloud_Left_CV", output_directory=output_dir())
            save_point_cloud(wide_pointcloud, "PointCloud_Wide_CV", output_directory=output_dir())
            save_point_cloud(combined_pointcloud, "PointCloud_Combined", output_directory=output_dir())
            cv.imwrite(path.join(output_dir(), "Depth_Narrow.png"), depth_narrow_colorized)
            cv.imwrite(path.join(output_dir(), "Depth_Wide.png"), depth_wide_colorized)

            # Save full point clouds
            save_point_cloud(full_narrow_point_cloud, "PointCloud_Left_CV_FULL", output_directory=output_dir())
            save_point_cloud(full_wide_point_cloud, "PointCloud_Wide_CV_FULL", output_directory=output_dir())
            cv.imwrite(path.join(output_dir(), "Depth_Narrow_FULL.png"), full_depth_narrow_colorized)
            cv.imwrite(path.join(output_dir(), "Depth_Wide_FULL.png"), full_depth_wide_colorized)

            # Save native point cloud
            save_point_cloud(native_point_cloud, "PointCloud_NATIVE_FULL", output_directory=output_dir())


    cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Implementation approach working on a captured snapshot (see capture.py)")
    parser.add_argument("directory", help="Path to the directory created by capture.py")
    args = parser.parse_args()
    main(args)
