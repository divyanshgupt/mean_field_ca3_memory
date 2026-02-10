import numpy as np


def response(W, I, ret_det = False):
    """Compute the linear response of a 3-population network.
    assumes that cellular gains are 1."""

    w_ll, w_le, w_lp = W[0, 0], W[0, 1], - W[0, 2]
    w_el, w_ee, w_ep = W[1, 0], W[1, 1], - W[1, 2]
    w_pl, w_pe, w_pp = W[2, 0], W[2, 1], - W[2, 2]

    dI_l, dI_e, dI_p = I[0], I[1], I[2]

    det = np.linalg.det(np.eye(3) - W)
    if det == 0:
        raise ValueError("The matrix (I - W) is singular, cannot compute response.")
    
    # det = 1
    R_l = (1/det) * (dI_l*(1 + w_pp - w_ee - w_ee*w_pp + w_ep*w_pe) + dI_e*(w_le + w_le*w_pp - w_lp*w_pe) + dI_p*(- w_lp - w_le*w_ep + w_lp*w_ee))

    R_e = (1/det) * (dI_l*(w_el + w_el*w_pp - w_ep*w_pl) + dI_e*(1 + w_pp - w_ll - w_ll*w_pp + w_lp*w_pl) + dI_p * (- w_ep - w_el*w_lp + w_ep*w_ll))

    if ret_det:
        return R_l, R_e, det
    else:
        return R_l, R_e


def response_regime_metric(R_l, R_e):
    """ returns the matrix of response regimes:
    0: both positive
    1: late positive, early negative or zero
    2: late negative or zero, early positive
    """

    regime = np.zeros(R_l.shape)
    regime[(R_l > 0) & (R_e > 0)] = 0
    regime[(R_l > 0) & (R_e <= 0)] = 1
    regime[(R_l <= 0) & (R_e > 0)] = 2
    regime[(R_l < 0) & (R_e < 0)] = 3
    return regime