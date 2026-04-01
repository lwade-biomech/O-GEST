# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 11:28:50 2025

@author: lw2175
"""

# main.py
import numpy as np

from trend_finder import trend_finder
from gest_one_landmark import gest_one_landmark
from events_finder import events_finder
from spatiotemporal_calculator import spatio_temporal_calculator
from visualisation_gest import visualization_gest
import ezc3d
=OK, all the code should now be done, now it is time to use antigravity and gemini CLI to try and get the whole code base working.
def o_gest(time, joints_L, joints_R, setting):
    #%%
    """
    Parameters
    ----------
    time : (N,) array
    joints_L : (N, M) array   # left foot trajectories (1, 2, or 3 landmarks)
    joints_R : (N, M) array   # right foot trajectories
    setting : dict            # {"visualization": True/False, "optimizer": "QP" or "SQP", ...}
    """

    visualization = setting.get("visualization", False)

    # =========================================================
    # Convert units if necessary (mm → m)
    # =========================================================
    if np.max(joints_L[:, 0]) - np.min(joints_L[:, 0]) > 20:
        joints_L = joints_L / 1000.0
        joints_R = joints_R / 1000.0


    # =========================================================
    # Trend & direction correction
    # =========================================================
    setting = trend_finder(time, joints_L, joints_R, setting)

    # Shift signals if needed
    if setting["type"] == "without_intersection":
        if setting["lower"] == "left":
            joints_L = joints_L + setting["shift_offset"]
        else:
            joints_R = joints_R + setting["shift_offset"]

    # Flip signals
    if setting["direction"] == "Upward":
        joints_L = -joints_L
        joints_R = -joints_R

    # =========================================================
    # Model selection based on landmark count
    # =========================================================
    M = joints_L.shape[1]   # number of landmarks

    info_L = [None] * M
    info_R = [None] * M

    for i in range(M):
        print(f"Optimizing landmark {i+1}/{M}...")
        info_L[i], info_R[i] = gest_one_landmark(
            time,
            joints_L[:, i],
            joints_R[:, i],
            setting
        )

    # =========================================================
    # Event detection
    # =========================================================
    events_location, events_time = events_finder(info_L, info_R, setting)


    # =========================================================
    # Correct INFO (trend, type)
    # =========================================================
    # Placeholder — to be implemented
    # info_L = [trend_info_corrector(x, setting) for x in info_L]
    # ...

    # =========================================================
    # Visualization
    # =========================================================
    visualization_gest(info_L, info_R)
    # =========================================================
    # Compute spatiotemporal parameters
    # =========================================================
    spatio = spatio_temporal_calculator(events_time)

    # =========================================================
    # Output structure
    # =========================================================
    events = {
        "time": events_time,
        "location": events_location
    }

    info = {
        "left": info_L,
        "right": info_R
    }

    #%%
    return events, spatio, info


if __name__ == "__main__":
    print("This file defines the main O-GEST entry point.")
    print('If this script is called directly it will use example data')


    file = '../Example_Python/Example_Gait.c3d'

    c3d = ezc3d.c3d(file)

    # --- Markers ---
    points = c3d['data']['points']  # shape: (4, n_markers, n_frames)
    marker_names = c3d['parameters']['POINT']['LABELS']['value']
    n_markers = points.shape[1]
    n_frames = points.shape[2]

    markers = {}
    setting = {}
    for i, name in enumerate(marker_names):
        # Ignore the 4th row (residual/quality) and transpose to (n_frames, 3)
        markers[name] = points[:3, i, :].T


    Toe_L_Horizontal = markers['LTOE'][:, 1]
    Toe_R_Horizontal = markers['RTOE'][:, 1]

    Heel_L_Horizontal = markers['LHEE'][:, 1]
    Heel_R_Horizontal = markers['RHEE'][:, 1]

    Ankle_L_Horizontal = markers['LANK'][:, 1]
    Ankle_R_Horizontal = markers['RANK'][:, 1]

    joints_L = np.column_stack((Toe_L_Horizontal, Heel_L_Horizontal, Ankle_L_Horizontal))
    joints_R = np.column_stack((Toe_R_Horizontal, Heel_R_Horizontal, Ankle_R_Horizontal))

    # --- Time vector ---
    frame_rate = c3d['parameters']['POINT']['RATE']['value'][0]  # frames per second
    time = np.arange(n_frames) / frame_rate  # time in seconds

    setting['visualization'] = "ON" 
    
    events, spatio, info = o_gest(time, joints_L, joints_R, setting)
    