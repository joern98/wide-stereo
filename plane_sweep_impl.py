import numpy as np
import cython
import math

# TODO only write the plane consistency computation in cython for now
#  since that is the only part where we cannot use fast OpenCV or numpy implementation

# both uint8 and float32 are compatible with C
# https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.uint8
SRC_DTYPE = np.uint8
DST_DTYPE = np.float32


@cython.cfunc
@cython.exceptvar(check=False)
@cython.boundscheck(False)  # we check bounds manually
@cython.wraparound(False)  # we don't need wraparound (negative indices)
def _compute_ncc(ref: np.ndarray, src: np.ndarray, x: cython.Py_ssize_t, y: cython.Py_ssize_t, window_size: cython.int):
    radius: cython.int = window_size // 2
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

    n_window = window_size * window_size
    mean_ref: cython.float = sum_ref / n_window
    mean_src: cython.float = sum_src / n_window

    # I could use Fast inverse square here for standard deviations
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
            sum_ref_1 += a*a
            sum_src_1 += b*b
            sum_numerator += a*b

    std_ref: cython.float = math.sqrt(sum_ref_1)
    std_src: cython.float = math.sqrt(sum_src_1)
    p = std_ref * std_src
    if p == 0:
        return 0
    ncc = sum_numerator / p
    return ncc


# TODO memoryviews
# TODO cfunc for the main part
def compute_consistency_image(ref: np.ndarray, src: np.ndarray, dst: np.ndarray,
                              window_size: cython.int = 3):
    """
    Compute Consistency of reference image with n source images using normalized cross-correlation

    :param ref: Reference Image with shape (h, w) and dtype uint8
    :param src: Array of n source images with shape (n, h, w) and dtype uint8
    :param dst: Destination Array with shape (h, w) and dtype float32
    :param window_size: search window size, has to be an odd number > 1, default 3
    :return: dst
    """
    assert ref.shape == src.shape[1:]
    assert dst.shape == ref.shape
    assert ref.dtype == SRC_DTYPE
    assert src.dtype == SRC_DTYPE
    assert dst.dtype == DST_DTYPE

    # analogous to width, height
    x_max: cython.Py_ssize_t = ref.shape[0]
    y_max: cython.Py_ssize_t = ref.shape[1]
    n_src: cython.Py_ssize_t = src.shape[0]

    x: cython.Py_ssize_t
    y: cython.Py_ssize_t
    k: cython.Py_ssize_t
    c: cython.float
    for x in range(x_max):
        for y in range(y_max):
            c = 0.0
            for k in range(n_src):
                c = c + _compute_ncc(ref, src[k], x, y, window_size)
            c /= n_src  # mean of computed NCC values
            dst[x, y] = c
    return dst