#cython: infer_types=True
#cython: boundscheck=False

from libc.math cimport sqrt, cos, sin, atan2

import numpy as np
cimport numpy as cnp

from cython.parallel import parallel, prange

# based on https://github.com/jtauber/mandelbulb/

cdef cnp.uint8_t mandel(int n, float x0, float y0, float z0) nogil:
    cdef float ct, st, cp, sp, rn
    cdef float r, theta, phi, xsqr, ysqr
    cdef int i

    x, y, z = 0.0, 0.0, 0.0
    for i in range(32):
        xsqr = x * x
        ysqr = y * y
        r = sqrt(xsqr + ysqr + z*z)
        theta = atan2(sqrt(xsqr + ysqr), z)
        phi = atan2(y, x)

        ct = cos(theta * n)
        st = sin(theta * n)
        cp = cos(phi * n)
        sp = sin(phi * n)
        rn = r ** n

        x = rn * st * cp + x0
        y = rn * st * sp + y0
        z = rn * ct + z0

        if (x**2 + y**2 + z**2) > 2:
            return 256 - (i * 4)

    return 0


def mandelbrot_pixel(float x, float y, float z, *,
                     int image_width=800,
                     int image_height=800,
                     scale=None, int n=8):
    if scale is None:
        scale = (4.0, 4.0, 50.0)

    x_scale, y_scale, z_scale = scale
    return mandel(n,
                  x_scale * (x - (image_width / 2.0)) / image_width,
                  y_scale * (y - (image_height / 2.0)) / image_height,
                  z / z_scale)


def mandelbrot_image(int z, int image_width=800, int image_height=800, *,
                     scale=None, int n=8):
    cdef float z_pos
    cdef int i
    if scale is None:
        scale = (4.0, 4.0, 50.0)

    x_pos = np.array([x for x in range(image_width) for y in range(image_height)],
                     dtype=np.float32)
    y_pos = np.array([y for x in range(image_width) for y in range(image_height)],
                     dtype=np.float32)

    x_scale, y_scale, z_scale = scale
    x_pos = x_scale * (x_pos - (image_width / 2.0)) / image_width
    y_pos = y_scale * (y_pos - (image_height / 2.0)) / image_height
    z_pos = z / z_scale

    cdef float [:] xpos_view = x_pos
    cdef float [:] ypos_view = y_pos
    cdef int num_points = len(x_pos)
    cdef cnp.uint8_t[:] data = np.zeros(num_points, dtype=np.uint8)

    for i in prange(num_points, nogil=True):
        data[i] = mandel(n, xpos_view[i], ypos_view[i], z_pos)

    arr = np.asarray(data).reshape(image_width, image_height)
    return np.minimum(255, np.maximum(0, arr))
