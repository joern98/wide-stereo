import json
import threading
import time
from dataclasses import dataclass, fields
from datetime import datetime
from json import JSONEncoder
from typing import Tuple, Sequence

import cv2 as cv
import numpy as np
import pyrealsense2 as rs

from device_utility.DevicePair import DevicePair
from device_utility.utils import set_sensor_option, get_stereo_extrinsic

NUM_PATTERNS_REQUIRED = 10
# https://docs.opencv.org/4.x/d9/d5d/classcv_1_1TermCriteria.html, (TYPE, iterations, epsilon)
TERM_CRITERIA = (cv.TERM_CRITERIA_COUNT + cv.TERM_CRITERIA_EPS, 30, 0.001)
WINDOW_IMAGE_LEFT = "left ir"
WINDOW_IMAGE_RIGHT = "right ir"


# cv.UMat is np.ndarray internally
@dataclass()
class CalibrationResult:
    # return values: retval (rms), cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F, perViewErrors
    # tuple[float, UMat, UMat, UMat, UMat, UMat, UMat, UMat, UMat, UMat
    rms: float
    camera_matrix_left: np.ndarray
    coeffs_left: np.ndarray
    camera_matrix_right: np.ndarray
    coeffs_right: np.ndarray
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    per_view_errors: np.ndarray
    R_14: np.ndarray | None  # optional 4x4 transformation matrix from outer left to outer right
    image_size: Tuple[int, int]


@dataclass()
class CameraParameters:
    left_rs_intrinsics: rs.intrinsics
    left_camera_matrix: np.ndarray
    left_dist_coeffs: np.array
    right_rs_intrinsics: rs.intrinsics
    right_camera_matrix: np.ndarray
    right_dist_coeffs: np.array
    image_size: Tuple[int, int]

    # left camera stereo extrinsic [left IR(1) -> right IR(2)]
    left_stereo_extrinsics: rs.extrinsics

    # right camera stereo extrinsic [left IR(1) -> right IR(2)]
    right_stereo_extrinsics: rs.extrinsics


@dataclass()
class RectificationResult:
    left_map_x: np.ndarray
    left_map_y: np.ndarray
    right_map_x: np.ndarray
    right_map_y: np.ndarray
    R_left: np.ndarray
    R_right: np.ndarray
    P_left: np.ndarray
    P_right: np.ndarray
    Q: np.ndarray
    ROI_left: Sequence[int]
    ROI_right: Sequence[int]


def run_camera_calibration(device_pair: DevicePair) -> Tuple[CalibrationResult, RectificationResult]:
    cv.namedWindow(WINDOW_IMAGE_RIGHT)
    cv.namedWindow(WINDOW_IMAGE_LEFT)

    # we only need IR streams
    # try lower resolution to improve performance
    device_pair.start(width=1280, height=720, fps=30, streams=(rs.stream.infrared,))

    # inner cameras
    camera_parameters = collect_camera_parameters(device_pair, 2, 1)

    object_points, image_points_left, image_points_right = find_chessboard_corners(device_pair)

    calibration_result = stereo_calibrate(device_pair, camera_parameters, object_points, image_points_left,
                                          image_points_right)

    # transform calibration
    calibration_result.R_14 = transform_inner_to_outer_stereo(camera_parameters, calibration_result)

    rectification_result = stereo_rectify(device_pair, camera_parameters.image_size, calibration_result)

    device_pair.stop()
    cv.destroyAllWindows()

    # TODO do not return calibration result as is, only return parameters for outer cameras

    write_calibration_to_file(calibration_result)

    return calibration_result, rectification_result


def collect_camera_parameters(device_pair: DevicePair, left_ir_index=1, right_ir_index=2) -> CameraParameters:
    # inner IR cameras
    left_intrinsic: rs.intrinsics = device_pair.left.pipeline_profile.get_stream(rs.stream.infrared, left_ir_index) \
        .as_video_stream_profile().get_intrinsics()
    right_intrinsic: rs.intrinsics = device_pair.right.pipeline_profile.get_stream(rs.stream.infrared, right_ir_index) \
        .as_video_stream_profile().get_intrinsics()

    left_camera_matrix = rs_intrinsics_to_camera_matrix(left_intrinsic)
    right_camera_matrix = rs_intrinsics_to_camera_matrix(right_intrinsic)

    left_coefficients = np.array(left_intrinsic.coeffs).astype(np.float32)
    right_coefficients = np.array(right_intrinsic.coeffs).astype(np.float32)

    left_stereo_extrinsic = get_stereo_extrinsic(device_pair.left.pipeline_profile)
    right_stereo_extrinsic = get_stereo_extrinsic(device_pair.right.pipeline_profile)

    camera_params = CameraParameters(left_rs_intrinsics=left_intrinsic,
                                     left_camera_matrix=left_camera_matrix,
                                     left_dist_coeffs=left_coefficients,
                                     left_stereo_extrinsics=left_stereo_extrinsic,
                                     right_rs_intrinsics=right_intrinsic,
                                     right_camera_matrix=right_camera_matrix,
                                     right_dist_coeffs=right_coefficients,
                                     right_stereo_extrinsics=right_stereo_extrinsic,
                                     image_size=(left_intrinsic.width, left_intrinsic.height))
    return camera_params


# https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html
def find_chessboard_corners(device_pair: DevicePair):
    # turn of ir pattern emitters
    depth_sensor_left: rs.depth_sensor = device_pair.left.device.first_depth_sensor()
    depth_sensor_right: rs.depth_sensor = device_pair.right.device.first_depth_sensor()
    set_sensor_option(depth_sensor_left, rs.option.emitter_enabled, False)
    set_sensor_option(depth_sensor_right, rs.option.emitter_enabled, False)

    # set exposure to below 33ms to allow for 30fps streaming
    set_sensor_option(depth_sensor_left, rs.option.exposure, 30000)
    set_sensor_option(depth_sensor_right, rs.option.exposure, 30000)

    # Initialize array to hold the 3D-object coordinates of the inner chessboard corners
    # 8x8 chessboard has 7x7 inner corners
    # objp = np.zeros((7 * 7, 3), np.float32)
    objp = np.zeros((5 * 7, 3), np.float32)

    # create coordinate pairs for the corners and write them to the array, leaving the z-coordinate at 0
    # chessboard pattern has a size of 24mm -> 0.024m
    # objp[:, :2] = np.mgrid[0:7 * 0.024:0.024, 0:7 * 0.024:0.024].T.reshape(-1, 2)
    objp[:, :2] = np.mgrid[0:5 * 0.034:0.034, 0:7 * 0.034:0.034].T.reshape(-1, 2)  # pattern size 34mm, 5x7

    object_points = []
    image_points_left = []
    image_points_right = []

    cooldown = False

    def reset_cooldown():
        nonlocal cooldown
        cooldown = False

    t0 = time.perf_counter_ns()
    # get chessboard corners until required number of valid correspondences has been found
    while np.size(object_points, 0) < NUM_PATTERNS_REQUIRED + 1:
        frame_left, frame_right = device_pair.wait_for_frames()

        # check frame timestamps
        ts_l = frame_left.get_timestamp()
        ts_r = frame_right.get_timestamp()
        d_ts = abs(ts_l - ts_r)
        print(f"d ts: {d_ts}")

        # outer cameras
        # ir_left = frame_left.get_infrared_frame(1)
        # ir_right = frame_right.get_infrared_frame(2)

        # inner cameras
        ir_left = frame_left.get_infrared_frame(2)
        ir_right = frame_right.get_infrared_frame(1)
        image_left = np.array(ir_left.get_data())
        image_right = np.array(ir_right.get_data())

        # find chessboard corners in both images. FAST_CHECK flag shortcuts the call if no chessboard is found
        # ret_l, corners_left = cv.findChessboardCorners(image_left, (7, 7), flags=cv.CALIB_CB_FAST_CHECK)
        # ret_r, corners_right = cv.findChessboardCorners(image_right, (7, 7), flags=cv.CALIB_CB_FAST_CHECK)

        # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gadc5bcb05cb21cf1e50963df26986d7c9
        # this is rather slow, maybe try multiprocessing or multithreading
        ret_l, corners_left = cv.findChessboardCornersSB(image_left, (5, 7), flags=cv.CALIB_CB_MARKER)
        ret_r, corners_right = cv.findChessboardCornersSB(image_right, (5, 7), flags=cv.CALIB_CB_MARKER)

        # if both images had valid chessboard patterns found, refine them and append them to the output array
        if ret_l and ret_r and not cooldown:
            object_points.append(objp)  # corresponding object points

            # corners_subpixel_left = cv.cornerSubPix(image_left, corners_left, (5, 5), (-1, -1), TERM_CRITERIA)
            # corners_subpixel_right = cv.cornerSubPix(image_right, corners_right, (5, 5), (-1, -1), TERM_CRITERIA)

            # image_points_left.append(corners_subpixel_left)
            # image_points_right.append(corners_subpixel_right)

            image_points_left.append(corners_left)
            image_points_right.append(corners_right)

            # draw corners on images
            # cv.drawChessboardCorners(image_left, (7, 7), corners_subpixel_left, ret_l)
            # cv.drawChessboardCorners(image_right, (7, 7), corners_subpixel_right, ret_r)
            cv.drawChessboardCorners(image_left, (5, 7), corners_left, ret_l)
            cv.drawChessboardCorners(image_right, (5, 7), corners_right, ret_r)

            # set cooldown period
            cooldown = True
            threading.Timer(2, reset_cooldown).start()

        # TODO refactor to use existing window infrastructure
        cv.imshow(WINDOW_IMAGE_LEFT, image_left)
        cv.imshow(WINDOW_IMAGE_RIGHT, image_right)

        t1 = time.perf_counter_ns()
        td = t1 - t0
        t0 = t1
        print(f"td: {td / 1000000:0.4} ms")
        if cv.waitKey(1) == 27:  # ESCAPE
            print(f"chessboard corner process aborted, found {np.size(object_points, 0)} sets of correspondences")
            break

    # turn emitters back on
    set_sensor_option(depth_sensor_left, rs.option.emitter_enabled, True)
    set_sensor_option(depth_sensor_right, rs.option.emitter_enabled, True)

    # discard first match, it is usually bad due to motion blur
    object_points.pop(0)
    image_points_left.pop(0)
    image_points_right.pop(0)

    return object_points, image_points_left, image_points_right


def rs_intrinsics_to_camera_matrix(intrinsics: rs.intrinsics) -> np.ndarray:
    m = np.zeros((3, 3), np.float32)
    m[0, 0] = intrinsics.fx
    m[1, 1] = intrinsics.fy
    m[0, 2] = intrinsics.ppx
    m[1, 2] = intrinsics.ppy
    m[2, 2] = 1
    return m


# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga9d2539c1ebcda647487a616bdf0fc716
def stereo_calibrate(device_pair: DevicePair, camera_params: CameraParameters, object_points, image_points_left,
                     image_points_right):
    # parameters: cv.stereoCalibrate(	objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize,...)
    # return values: retval (rms), cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F, perViewErrors
    per_view_errors = np.zeros(np.size(object_points, 0), np.float32)
    r = np.zeros((3, 3), np.float32)
    t = np.zeros((3, 1), np.float32)
    result = cv.stereoCalibrate(objectPoints=object_points,
                                imagePoints1=image_points_left,
                                imagePoints2=image_points_right,
                                cameraMatrix1=camera_params.left_camera_matrix,
                                distCoeffs1=camera_params.left_dist_coeffs,
                                cameraMatrix2=camera_params.right_camera_matrix,
                                distCoeffs2=camera_params.right_dist_coeffs,
                                imageSize=camera_params.image_size,
                                R=r,
                                T=t,
                                perViewErrors=per_view_errors,
                                flags=cv.CALIB_FIX_INTRINSIC)

    # set R_14 if its outer pair
    calibration_result = CalibrationResult(*result, R_14=None, image_size=camera_params.image_size)
    print(f"stereo calibration rms: {calibration_result.rms}")
    print(f"stereo calibration per view errors: \n{calibration_result.per_view_errors}")
    return calibration_result


def transform_inner_to_outer_stereo(camera_params: CameraParameters, calib: CalibrationResult):
    """
    Return the 4x4 transformation matrix R_14=(R|t) in homogenous coordinates
    :param camera_params:
    :param calib: Calibration result from calibrating the inner cameras of the device pair
    :return:
    """
    # R14 = R34*R23*R12
    # R12 and R34 are camera extrinsic parameters
    # R23 is calibration result
    R_12 = np.eye(4, dtype=np.float32)
    R_23 = np.eye(4, dtype=np.float32)
    R_34 = np.eye(4, dtype=np.float32)

    # rs.extrinsics.rotation is column-major 3x3 matrix -> transpose to row major for compatibility with openCV
    R_12[:3, :3] = np.asarray(camera_params.left_stereo_extrinsics.rotation).reshape(3, 3).T
    R_12[:3, 3:4] = np.asarray(camera_params.left_stereo_extrinsics.translation).reshape(3, 1)

    R_34[:3, :3] = np.asarray(camera_params.right_stereo_extrinsics.rotation).reshape(3, 3).T
    R_34[:3, 3:4] = np.asarray(camera_params.right_stereo_extrinsics.translation).reshape(3, 1)

    # calib.R is already row-major as it was created by openCV
    R_23[:3, :3] = calib.R
    R_23[:3, 3:4] = calib.T

    # @ is shorthand for np.matmul(a, b)
    R_14 = R_34 @ R_23 @ R_12
    return R_14


# Reference: https://learnopencv.com/making-a-low-cost-stereo-camera-using-opencv/#stereo-rectification
# https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4
def stereo_rectify(device_pair: DevicePair, image_size: Tuple[int, int], calib: CalibrationResult):
    # transform inner camera calibration to outer camera calibration transform
    if calib.R_14 is not None:
        R = calib.R_14[:3, :3].astype(np.float64)
        T = calib.R_14[:3, 3:4].astype(np.float64)
    else:
        R = calib.R
        T = calib.T

    # cv.stereoRectify(	cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T[, R1[, R2[, P1[, P2[, Q[, flags[, alpha[, newImageSize]]]]]]]]	)
    # -> 	R1, R2, P1, P2, Q, validPixROI1, validPixROI2
    # All the matrices must have the same data type in function 'cvRodrigues2' -> convert to np.float32
    # camera params are the same as calib result
    # https://answers.opencv.org/question/3441/strange-stereorectify-error-with-rotation-matrix/ -> double precision

    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(cameraMatrix1=calib.camera_matrix_left,
                                                     distCoeffs1=calib.coeffs_left,
                                                     cameraMatrix2=calib.camera_matrix_right,
                                                     distCoeffs2=calib.coeffs_right,
                                                     imageSize=(1280, 720),  # fix hardcoded size
                                                     R=R,
                                                     T=T,
                                                     alpha=1)
    # cv.initUndistortRectifyMap(	cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type[, map1[, map2]]	) -> 	map1, map2
    # research data type to use for "m1type", 16SC2 -> 16 bit signed integer, two-channel, same as in reference
    # map_1 -> map_x, map_2 -> map_y
    left_map_1, left_map_2 = cv.initUndistortRectifyMap(cameraMatrix=calib.camera_matrix_left,
                                                        distCoeffs=calib.coeffs_left,
                                                        R=R1,
                                                        newCameraMatrix=P1,
                                                        size=image_size,
                                                        m1type=cv.CV_16SC2)
    right_map_1, right_map_2 = cv.initUndistortRectifyMap(cameraMatrix=calib.camera_matrix_right,
                                                          distCoeffs=calib.coeffs_right,
                                                          R=R1,
                                                          newCameraMatrix=P1,
                                                          size=image_size,
                                                          m1type=cv.CV_16SC2)

    rectification_result = RectificationResult(left_map_1,
                                               left_map_2,
                                               right_map_1,
                                               right_map_2,
                                               R1, R2, P1, P2, Q,
                                               roi1, roi2)
    return rectification_result


class CalibrationResultEncoder(JSONEncoder):

    def __serialize_calibration_result(self, obj: CalibrationResult):
        o = {
            "rms": obj.rms,
            "camera_matrix_left": obj.camera_matrix_left.tolist(),
            "distortion_coefficients_left": obj.coeffs_left.ravel().tolist(),
            "camera_matrix_right": obj.camera_matrix_right.tolist(),
            "distortion_coefficients_right": obj.coeffs_right.ravel().tolist(),
            "R": obj.R.tolist(),
            "T": obj.T.ravel().tolist(),
            "E": obj.E.tolist(),
            "F": obj.F.tolist(),
            "per_view_errors": obj.per_view_errors.tolist(),
            "R_14": obj.R_14.tolist(),
            "image_size": obj.image_size
        }
        return o

    def default(self, obj):
        if isinstance(obj, CalibrationResult):
            return self.__serialize_calibration_result(obj)
        return json.JSONEncoder.default(self, obj)


def write_calibration_to_file(calibration_result: CalibrationResult):
    filename = f"Calibration_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    with open(filename + ".json", "x") as f:
        json.dump(calibration_result, f, cls=CalibrationResultEncoder, indent=2)
        print(f"Written human-readable calibration data to file: {filename + '.json'}")

    with open(filename + ".npy", "xb") as f:
        # order matters, save all fields of Calibration result
        for field in fields(CalibrationResult):
            np.save(f, getattr(calibration_result, field.name))
        print(f"Written binary calibration data to file: {filename + '.npy'}")


def load_calibration_from_file(filename: str) -> CalibrationResult:
    with open(filename, "rb") as f:
        # this is hacky, iterate through the fields and load the data in that order from the .npy file
        calibration_result = CalibrationResult(None, None, None, None, None, None, None, None, None, None, None, None)
        for field in fields(CalibrationResult):
            value = np.load(f)
            setattr(calibration_result, field.name, value)

    print(f"Calibration loaded from {filename}:\n{calibration_result}")
    return calibration_result
