import numpy as np
from scipy.interpolate import CubicSpline


def pixel_soh(XPred):
    Pred_SOH = []
    Cap = []
    for i in range(256):
        y = XPred[:, i]
        y_coord = np.argmin(y)
        cap_value = (1 - (y_coord + 1) / 256) * 0.16 + 0.8
        Cap.append(cap_value)

    x = np.arange(1, 257)
    y = np.array(Cap)

    cs = CubicSpline(x, y)

    x1 = np.linspace(1, 256, 60)
    y1 = cs(x1)

    Pred_SOH.append(y1)
    Pred_SOH = np.array(Pred_SOH)
    Pred_SOH = Pred_SOH[np.lexsort((Pred_SOH[:, 1], Pred_SOH[:, 0]))]

    return Pred_SOH
