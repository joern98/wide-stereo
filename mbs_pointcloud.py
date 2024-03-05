import argparse
import json
import os.path
import os.path as path
from datetime import datetime

import cv2 as cv
import numpy as np
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

MAP_DEPTH_M_TO_BYTE = interp1d([0, 10], [0, 255], bounds_error=False, fill_value=(0, 255))

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
                         colormap: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    if colormap is not None:
        depth_image = o3d.geometry.Image(depth)
        # OpenCV is BGR, Open3D expects RGB
        depth_colormap_image = o3d.geometry.Image(colormap)
        depth_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=depth_colormap_image,
                                                                              depth=depth_image,
                                                                              depth_trunc=20,
                                                                              depth_scale=1,
                                                                              convert_rgb_to_intensity=False)
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(depth_rgbd_image, intrinsic, extrinsic)
    else:
        depth_image = o3d.geometry.Image(depth)
        pc = o3d.geometry.PointCloud.create_from_depth_image(depth_image,
                                                             intrinsic,
                                                             extrinsic,
                                                             depth_scale=1,
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
        pathname = path.join(root_directory, f"Output_{timestamp}")
        os.mkdir(pathname)
        OUTPUT_DIRECTORY = pathname
    return OUTPUT_DIRECTORY


def main(args):
    calibration_result, camera_parameters, \
        left_ir_1, left_ir_2, right_ir_1, right_ir_2, \
        left_native_depth, right_native_depth = load_data(args.directory)

    # show raw infrared images
    cv.imshow(WINDOW_LEFT_IR_1, left_ir_1)
    cv.imshow(WINDOW_LEFT_IR_2, left_ir_2)
    cv.imshow(WINDOW_RIGHT_IR_1, right_ir_1)
    cv.imshow(WINDOW_RIGHT_IR_2, right_ir_2)

    stereo95 = cv.StereoSGBM.create(
        minDisparity=1,
        numDisparities=16 * 4,
        blockSize=3,
        P1=100,
        P2=400,
        disp12MaxDiff=4,
        preFilterCap=1,
        uniquenessRatio=5,
        speckleWindowSize=64,
        speckleRange=1,
        mode=cv.STEREO_SGBM_MODE_SGBM
    )

    # compute stereo left device
    disp95 = stereo95.compute(left_ir_1, left_ir_2).astype(np.float32) / 16.0
    b95 = abs(camera_parameters["left_stereo_extrinsics"]["t"][0])
    f95 = camera_parameters["left_intrinsics"]["fx"]
    bf95 = b95 * f95
    depth95 = bf95 / disp95

    depth95_color = cv.applyColorMap(MAP_DEPTH_M_TO_BYTE(depth95).astype(np.uint8), cv.COLORMAP_JET)
    cv.imshow(WINDOW_DEPTH_LEFT, depth95_color)

    stereo285 = cv.StereoSGBM.create(
        minDisparity=1,
        numDisparities=16 * 8,
        blockSize=3,
        P1=100,
        P2=400,
        disp12MaxDiff=4,
        preFilterCap=1,
        uniquenessRatio=5,
        speckleWindowSize=64,
        speckleRange=1,
        mode=cv.STEREO_SGBM_MODE_HH
    )

    # compute stereo wide
    # rectify first
    rectification = stereo_rectify(image_size=camera_parameters["image_size"], calib=calibration_result)
    left_rectified = cv.remap(left_ir_1,
                              rectification.left_map_x,
                              rectification.left_map_y,
                              interpolation=cv.INTER_LANCZOS4,
                              borderMode=cv.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
    right_rectified = cv.remap(right_ir_2,
                               rectification.right_map_x,
                               rectification.right_map_y,
                               interpolation=cv.INTER_LANCZOS4,
                               borderMode=cv.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))
    disp285 = stereo285.compute(left_rectified, right_rectified).astype(np.float32) / 16.0
    points285 = cv.reprojectImageTo3D(disp285, rectification.Q, handleMissingValues=True)
    depth285_color = cv.applyColorMap(
        MAP_DEPTH_M_TO_BYTE(points285).astype(np.uint8)[:, :, 2:3],
        cv.COLORMAP_JET)
    cv.imshow(WINDOW_DEPTH_WIDE, depth285_color)

    points285_flat = points285.reshape(-1, 3)
    invalid_indices = np.nonzero(points285_flat[:,
                                 2:3] == 10000)  # find indices where reprojectImageTo3D() has set Z to 10000 to mark invalid point
    points285_valid_only = np.delete(points285_flat, invalid_indices[0], 0)

    def on_mouse(event, x, y, flags, user_data):
        if event == cv.EVENT_MOUSEMOVE:
            print(
                f"left depth: {depth95[y, x]} m | wide depth: {points285[y, x, 2]} m | diff: {depth95[y, x] - points285[y, x, 2]} | left native "
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

    identity4 = np.eye(4)
    pinhole_intrinsics = create_pinhole_intrinsic_from_dict(camera_parameters["left_intrinsics"],
                                                            camera_parameters["image_size"])
    point_cloud95 = depth_to_point_cloud(depth95, pinhole_intrinsics, identity4, depth95_color)
    point_cloud285 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points285_valid_only.reshape(-1, 3)))
    # cv.cvtColor(depth285_color, cv.COLOR_BGR2RGB,
    #             dst=depth285_color)  # Open3D expects RGB [0, 1], OpenCV uses BGR [0, 255]
    # point_cloud285.colors = o3d.utility.Vector3dVector(depth285_color.reshape(-1, 3) / 0xFF)

    while run:
        key = cv.waitKey(1)
        if key == 27:  # ESCAPE
            run = False
        if key == ord('p'):
            save_point_cloud(point_cloud95, "PointCloud_Left_CV", output_directory=output_dir())
            save_point_cloud(point_cloud285, "PointCloud_Wide_CV", output_directory=output_dir())
        if key == ord('s'):
            cv.imwrite(path.join(output_dir(), "Depth_Narrow.png"), depth95_color)
            cv.imwrite(path.join(output_dir(), "Depth_Wide.png"), depth285_color)

    cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Implementation approach working on a captured snapshot (see capture.py)")
    parser.add_argument("directory", help="Path to the directory created by capture.py")
    args = parser.parse_args()
    main(args)
