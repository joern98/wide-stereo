import numpy as np
import cython


# Based on https://docs.cython.org/en/latest/src/userguide/numpy_tutorial.html

@cython.cfunc
@cython.exceptval(check=False)
@cython.boundscheck(False)  # we check bounds manually
@cython.wraparound(False)  # we don't need wraparound (negative indices)
def _compute_ncc(ref: cython.uchar[:, :],
                 src: cython.uchar[:, :],
                 x: cython.Py_ssize_t,
                 y: cython.Py_ssize_t,
                 window_size: cython.uint) -> cython.float:
    # Normalized Cross-Correlation
    # Furukawa, Hernandez: Multi-View Stereo, p.23; Szeliski: Computer Vision A&A, p. 448
    radius: cython.uint = window_size // 2
    if x - radius < 0 or y - radius < 0 or x + radius >= ref.shape[0] or y + radius >= ref.shape[1]:
        return 0

    # does this work with the types?, ssize_t cannot be negative
    min_x: cython.Py_ssize_t = x - radius
    min_y: cython.Py_ssize_t = y - radius
    d_window: cython.int = window_size

    u: cython.Py_ssize_t
    v: cython.Py_ssize_t
    sum_ref: cython.uint = 0
    sum_src: cython.uint = 0

    # compute mean over window
    for u in range(d_window):
        for v in range(d_window):
            sum_ref += ref[min_x + u, min_y + v]
            sum_src += src[min_x + u, min_y + v]

    n_window: cython.float = window_size * window_size
    mean_ref: cython.float = sum_ref / n_window
    mean_src: cython.float = sum_src / n_window

    # compute pseudo std. deviation over window
    sum_ref_1: cython.float = 0
    sum_src_1: cython.float = 0
    sum_numerator: cython.float = 0
    a: cython.float
    b: cython.float

    for u in range(d_window):
        for v in range(d_window):
            a = ref[min_x + u, min_y + v] - mean_ref
            b = src[min_x + u, min_y + v] - mean_src
            sum_ref_1 += a * a
            sum_src_1 += b * b
            sum_numerator += a * b

    if sum_src_1 == 0 or sum_ref_1 == 0:
        return 0

    # Inverse sqrt is faster than first computing sqrt and then taking the inverse
    std_ref: cython.float = sum_ref_1 ** -0.5
    std_src: cython.float = sum_src_1 ** -0.5
    p: cython.float = std_ref * std_src
    ncc: cython.float = sum_numerator * p
    return ncc


# cython memory views -> cython.uchar[:, :] (2D memoryview)
# https://docs.cython.org/en/latest/src/userguide/numpy_tutorial.html#efficient-indexing-with-memoryviews
@cython.boundscheck(False)
@cython.wraparound(False)
def compute_consistency_image(ref: cython.uchar[:, :], src: cython.uchar[:, :, :], dst: np.ndarray,
                              window_size: cython.int = 3):
    """
    Compute Consistency of reference image with n source images using normalized cross-correlation

    :param ref: Reference Image with shape (h, w) and dtype uint8
    :param src: Array of n source images with shape (n, h, w) and dtype uint8
    :param dst: Destination Array with shape (h, w) and dtype float32
    :param window_size: search window size, has to be an odd number > 1, default 3
    :return: dst
    """

    # since shape is now a simple C-Array, we can no longer do ref.shape == src.shape[1:]
    assert ref.shape[0] == src.shape[1] == dst.shape[0]
    assert ref.shape[1] == src.shape[2] == dst.shape[1]

    # image width, height
    x_max: cython.Py_ssize_t = ref.shape[0]
    y_max: cython.Py_ssize_t = ref.shape[1]
    n_src: cython.Py_ssize_t = src.shape[0]

    dst_view: cython.float[:, :] = dst

    x: cython.Py_ssize_t
    y: cython.Py_ssize_t
    k: cython.Py_ssize_t
    c: cython.float
    for y in range(y_max):
        for x in range(x_max):
            c = 0.0
            for k in range(n_src):
                c = c + _compute_ncc(ref, src[k], x, y, window_size)
            # for k in range(2):
            #    c = c + _compute_ncc(src[0], src[k+1], x, y, window_size)
            # c = c + _compute_ncc(src[1], src[2], x, y, window_size)
            # c /= n_src + 3  # mean of computed NCC values
            c /= n_src  # mean of computed NCC values
            dst_view[x, y] = c
    return dst
