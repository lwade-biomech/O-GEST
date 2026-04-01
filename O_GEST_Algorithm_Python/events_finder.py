# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 11:30:31 2025

@author: lw2175
"""

import numpy as np


import numpy as np

def events_finder(info_L_list, info_R_list, setting):
    """
    Find foot strike (GC) and foot off (SW) events from 3 landmarks per foot.

    Inputs:
        info_L_list : list of 3 dicts for left foot landmarks
        info_R_list : list of 3 dicts for right foot landmarks
        setting     : dict with key "Direction" = "Upward" or "Downward"

    Each dict must have 'Interests' key -> 2xN numpy array (time, location)

    Returns:
        Events_Location : dict with 2xN arrays
        Events_Time     : dict with 1D arrays
    """

    def slice_interests(info):
        if info is None or 'Interests' not in info or info['Interests'] is None:
            return np.empty((2,0))
        interests = info['Interests']
        ncols = interests.shape[1]
        GC = interests[:, 2:ncols:4] if ncols >= 3 else np.empty((2,0))
        SW = interests[:, 1:ncols:4] if ncols >= 2 else np.empty((2,0))
        return GC, SW

    def ensure_array(arr, width):
        if arr is None or arr.size == 0:
            return np.full((2,width), np.nan)
        return arr

    # Extract GC/SW for 3 landmarks per foot
    GC_L = []
    SW_L = []
    for i in range(3):
        GC, SW = slice_interests(info_L_list[i])
        GC_L.append(GC)
        SW_L.append(SW)
    GC_R = []
    SW_R = []
    for i in range(3):
        GC, SW = slice_interests(info_R_list[i])
        GC_R.append(GC)
        SW_R.append(SW)

    # Determine widths (fallback to first non-empty)
    width_L = next((gc.shape[1] for gc in GC_L if gc.size>0), 0)
    width_R = next((gc.shape[1] for gc in GC_R if gc.size>0), 0)

    # Fill empty arrays with NaNs if needed
    for i in range(3):
        GC_L[i] = ensure_array(GC_L[i], width_L)
        SW_L[i] = ensure_array(SW_L[i], width_L)
        GC_R[i] = ensure_array(GC_R[i], width_R)
        SW_R[i] = ensure_array(SW_R[i], width_R)

    # Select earliest GC (min) and latest SW (max) per foot
    GC_L_final = np.nanmin(np.stack(GC_L, axis=2), axis=2)
    GC_R_final = np.nanmin(np.stack(GC_R, axis=2), axis=2)
    SW_L_final = np.nanmax(np.stack(SW_L, axis=2), axis=2)
    SW_R_final = np.nanmax(np.stack(SW_R, axis=2), axis=2)

    # Build output
    Events_Time = {
        'Left_Foot_Strike': GC_L_final[0,:],
        'Right_Foot_Strike': GC_R_final[0,:],
        'Left_Foot_Off': SW_L_final[0,:],
        'Right_Foot_Off': SW_R_final[0,:]
    }

    Events_Location = {
        'Left_Foot_Strike': GC_L_final,
        'Right_Foot_Strike': GC_R_final,
        'Left_Foot_Off': SW_L_final,
        'Right_Foot_Off': SW_R_final
    }

    # Invert vertical if needed
    if setting.get("Direction") == "Upward":
        for k in Events_Location:
            Events_Location[k][1,:] *= -1

    return Events_Location, Events_Time




