import numpy as np
cimport numpy as np

cdef extern void joint_filter(float*, float*, float*, int, int, int, int, double, double)

def upsample(x_l, x_ab, s_x, s_l):
    n_up, m_up = x_l.shape[0], x_l.shape[1]
    n_dw, m_dw = x_ab.shape[1], x_ab.shape[2]

    cdef np.ndarray[float, ndim=2, mode="c"] l_up = np.ascontiguousarray(x_l)
    cdef np.ndarray[float, ndim=2, mode="c"] a_dw = np.ascontiguousarray(x_ab[0])
    cdef np.ndarray[float, ndim=2, mode="c"] b_dw = np.ascontiguousarray(x_ab[1])

    cdef np.ndarray[float, ndim=2, mode="c"] a_up = np.empty((n_up, m_up), dtype=np.float32)
    cdef np.ndarray[float, ndim=2, mode="c"] b_up = np.empty((n_up, m_up), dtype=np.float32)

    joint_filter(&l_up[0,0], &a_dw[0,0], &a_up[0,0], n_up, m_up, n_dw, m_dw, s_x, s_l)
    joint_filter(&l_up[0,0], &b_dw[0,0], &b_up[0,0], n_up, m_up, n_dw, m_dw, s_x, s_l)

    return np.stack([a_up, b_up], axis=0)


