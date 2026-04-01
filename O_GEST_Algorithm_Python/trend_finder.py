# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 11:29:51 2025

@author: lw2175
"""

import numpy as np

def trend_finder(time, joints_L, joints_R, setting):
    """   
    Parameters
    ----------
    time : (N,) array
    joints_L : (N, M) array
    joints_R : (N, M) array
    setting : dict-like (will be modified and returned)

    Returns
    -------
    setting : dict with keys:
        - "Direction": "Upward" or "Downward"
        - "Type": "Without_Intersection" or "With_Intersection"
        - "Shift_Offset": float (only if Type = Without_Intersection)
        - "Lower": "Left" or "Right" (only if Type = Without_Intersection)
    """

    time = np.asarray(time).flatten()
    joints_L = np.asarray(joints_L)
    joints_R = np.asarray(joints_R)

    # -------- 1. Compute slope of each Left and Right trajectory -------- #
    ML = np.array([
        np.polyfit(time, joints_L[:, i], 1)[0]
        for i in range(joints_L.shape[1])
    ])
    MR = np.array([
        np.polyfit(time, joints_R[:, i], 1)[0]
        for i in range(joints_R.shape[1])
    ])

    # -------- 2. Determine direction (Upwards vs Downwards) ------------- #
    # If all slopes > 0 → Reference is behind → Upward
    if np.all(ML > 0) and np.all(MR > 0):
        setting["direction"] = "upward"
    else:
        setting["direction"] = "downward"

    # -------- 3. Determine intersection type ---------------------------- #
    # Compare first landmark of Left vs Right
    diff = joints_L[1:-1, 0] - joints_R[1:-1, 0]

    neg_part = diff[diff < 0]
    pos_part = diff[diff > 0]

    no_neg = len(neg_part) == 0
    no_pos = len(pos_part) == 0

    # Ratio test
    if (no_neg or no_pos or
        (max(len(neg_part), len(pos_part)) /
         max(1, min(len(neg_part), len(pos_part)))) > 3):

        setting["type"] = "without_intersection"

        # Compute min and max abs differences → average
        min_diff = np.min(np.abs(diff))
        max_diff = np.max(np.abs(diff))
        setting["shift_offset"] = 0.5 * (min_diff + max_diff)

        # Determine which foot is "Lower"
        if joints_L[1, 0] < joints_R[1, 0]:
            setting["lower"] = "left"
        else:
            setting["lower"] = "right"

    else:
        setting["type"] = "With_Intersection"

    return setting
