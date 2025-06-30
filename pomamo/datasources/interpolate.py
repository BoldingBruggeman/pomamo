import numpy as np


def get_weights(i, j, ilow, jlow, mask=False, extrapolate_j=False):
    assert i >= ilow and i <= ilow + 1, f"target i={i}, lower point={ilow}"
    assert (
        j >= jlow and j <= jlow + 1 or extrapolate_j
    ), f"target j={j}, lower point={jlow}"
    if not np.any(mask):
        # We have all four corners - use bilinear interpolation
        wx2 = i - ilow
        wy2 = j - jlow
        wx1 = 1.0 - wx2
        wy1 = 1.0 - wy2
        w11 = wx1 * wy1
        w12 = wx1 * wy2
        w21 = wx2 * wy1
        w22 = wx2 * wy2
        weights = w11, w12, w21, w22
    else:
        # Some corners are masked
        # Use distance weighted N nearest neighbour interpolation
        assert len(mask) == 4
        assert not mask.all()
        dx1 = (i - ilow) ** 2
        dx2 = (ilow + 1 - i) ** 2
        dy1 = (j - jlow) ** 2
        dy2 = (jlow + 1 - j) ** 2
        w11 = 0.0 if mask[0] else 1.0 / np.sqrt(dx1 + dy1)
        w12 = 0.0 if mask[1] else 1.0 / np.sqrt(dx1 + dy2)
        w21 = 0.0 if mask[2] else 1.0 / np.sqrt(dx2 + dy1)
        w22 = 0.0 if mask[3] else 1.0 / np.sqrt(dx2 + dy2)
        w_tot = w11 + w12 + w21 + w22
        weights = w11 / w_tot, w12 / w_tot, w21 / w_tot, w22 / w_tot
    return weights
