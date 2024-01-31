import cv2 as cv
import numpy as np
from device_utility.DevicePair import DevicePair

NUM_PATTERNS_REQUIRED = 10
# https://docs.opencv.org/4.x/d9/d5d/classcv_1_1TermCriteria.html, (TYPE, iterations, epsilon)
TERM_CRITERIA = (cv.TERM_CRITERIA_COUNT + cv.TERM_CRITERIA_EPS, 30, 0.001)
WINDOW_IMAGE_LEFT = "left ir"
WINDOW_IMAGE_RIGHT = "right ir"


def run_camera_calibration(device_pair: DevicePair):
    device_pair.start()
    object_points, image_points_left, image_points_right = find_chessboard_corners(device_pair)
    print(object_points, image_points_left, image_points_right)
    device_pair.stop()
    pass


# https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html
def find_chessboard_corners(device_pair: DevicePair):

    # TODO turn of ir pattern emitters

    # Initialize array to hold the 3D-object coordinates of the inner chessboard corners
    # 8x8 chessboard has 7x7 inner corners
    objp = np.zeros((7 * 7, 3), np.float32)

    # create coordinate pairs for the corners and write them to the array, leaving the z-coordinate at 0
    # chessboard pattern has a size of 24mm -> 0.024m
    objp[:, :2] = np.mgrid[0:7 * 0.024:0.024, 0:7 * 0.024:0.024].T.reshape(-1, 2)

    object_points = []
    image_points_left = []
    image_points_right = []

    # get chessboard corners until required number of valid correspondences has been found
    while np.size(object_points, 0) < NUM_PATTERNS_REQUIRED:
        frame_left, frame_right = device_pair.wait_for_frames()
        ir_left = frame_left.get_infrared_frame(1)
        ir_right = frame_right.get_infrared_frame(2)
        image_left = np.array(ir_left.get_data())
        image_right = np.array(ir_right.get_data())

        # find chessboard corners in both images. FAST_CHECK flag shortcuts the call if no chessboard is found
        ret_l, corners_left = cv.findChessboardCorners(image_left, (7, 7), flags=cv.CALIB_CB_FAST_CHECK)
        ret_r, corners_right = cv.findChessboardCorners(image_right, (7, 7), flags=cv.CALIB_CB_FAST_CHECK)

        # if both images had valid chessboard patterns found, refine them and append them to the output array
        if ret_l and ret_r:
            object_points.append(objp)  # corresponding object points

            corners_subpixel_left = cv.cornerSubPix(image_left, corners_left, (5, 5), (-1, -1), TERM_CRITERIA)
            corners_subpixel_right = cv.cornerSubPix(image_right, corners_right, (5, 5), (-1, -1), TERM_CRITERIA)

            image_points_left.append(corners_subpixel_left)
            image_points_right.append(corners_subpixel_right)

            # show image for debugging purposes
            cv.drawChessboardCorners(image_left, (7, 7), corners_subpixel_left, ret_l)
            cv.imshow(WINDOW_IMAGE, image_left)
            cv.waitKey(0)

    # corners have been found
    return object_points, image_points_left, image_points_right





def stereo_calibrate(device_pair: DevicePair, object_points, image_points1, image_points2):
    pass
