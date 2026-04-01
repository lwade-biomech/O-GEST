# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 11:30:50 2025

@author: lw2175
"""

import numpy as np

def spatio_temporal_calculator(Events):
    """
    Python equivalent of Spatio_Temporal_Calculator.m
    Computes stride/step/swing/stance parameters for left and right foot.
    """

    try:
        Spatio = {}

        # ----- Extract event matrices -----
        LFS = Events["Location"]["Left_Foot_Strike"]      # shape (2, N)
        LFO = Events["Location"]["Left_Foot_Off"]
        RFS = Events["Location"]["Right_Foot_Strike"]
        RFO = Events["Location"]["Right_Foot_Off"]

        nL = LFS.shape[1] - 1
        nR = RFS.shape[1] - 1

        # ---------- LEFT FOOT ----------
        stride_len_L = np.zeros(nL)
        stride_time_L = np.zeros(nL)
        step_len_L = np.zeros(nL)
        step_time_L = np.zeros(nL)
        speed_L = np.zeros(nL)
        swing_L = np.zeros(nL)
        stance_L = np.zeros(nL)

        for i in range(nL):
            s1 = LFS[:, i]
            s2 = LFS[:, i + 1]

            stride_len_L[i] = abs(s2[1] - s1[1])
            stride_time_L[i] = abs(s2[0] - s1[0])
            speed_L[i] = stride_len_L[i] / stride_time_L[i]

            # --- Extract foot-off inside this stride window ---
            offs = LFO[:, (LFO[0] >= s1[0]) & (LFO[0] <= s2[0])]
            if offs.size > 0:
                swing_L[i] = 100 * abs(s2[0] - offs[0, 0]) / stride_time_L[i]
            else:
                swing_L[i] = np.nan

            stance_L[i] = 100 - swing_L[i]

            # --- Opposite strikes inside window ---
            opp = RFS[:, (RFS[0] >= s1[0]) & (RFS[0] <= s2[0])]
            if opp.size > 0:
                step_len_L[i] = abs(opp[1, 0] - s2[1])
                step_time_L[i] = abs(opp[0, 0] - s2[0])
            else:
                step_len_L[i] = np.nan
                step_time_L[i] = np.nan

        # ---------- RIGHT FOOT ----------
        stride_len_R = np.zeros(nR)
        stride_time_R = np.zeros(nR)
        step_len_R = np.zeros(nR)
        step_time_R = np.zeros(nR)
        speed_R = np.zeros(nR)
        swing_R = np.zeros(nR)
        stance_R = np.zeros(nR)

        for i in range(nR):
            s1 = RFS[:, i]
            s2 = RFS[:, i + 1]

            stride_len_R[i] = abs(s2[1] - s1[1])
            stride_time_R[i] = abs(s2[0] - s1[0])
            speed_R[i] = stride_len_R[i] / stride_time_R[i]

            offs = RFO[:, (RFO[0] >= s1[0]) & (RFO[0] <= s2[0])]
            if offs.size > 0:
                swing_R[i] = 100 * abs(s2[0] - offs[0, 0]) / stride_time_R[i]
            else:
                swing_R[i] = np.nan

            stance_R[i] = 100 - swing_R[i]

            opp = LFS[:, (LFS[0] >= s1[0]) & (LFS[0] <= s2[0])]
            if opp.size > 0:
                step_len_R[i] = abs(opp[1, 0] - s2[1])
                step_time_R[i] = abs(opp[0, 0] - s2[0])
            else:
                step_len_R[i] = np.nan
                step_time_R[i] = np.nan

        # ----- Overall -----
        Spatio["Overall_Gait_Speed"] = np.nanmean(np.concatenate([speed_L, speed_R]))
        Spatio["Overall_Step_Length"] = np.nanmean(np.concatenate([step_len_L, step_len_R]))
        Spatio["Overall_Step_Time"] = np.nanmean(np.concatenate([step_time_L, step_time_R]))
        Spatio["Overall_Stride_Length"] = np.nanmean(np.concatenate([stride_len_L, stride_len_R]))
        Spatio["Overall_Stride_Time"] = np.nanmean(np.concatenate([stride_time_L, stride_time_R]))
        Spatio["Overall_Stance_Percentage"] = np.nanmean(np.concatenate([stance_L, stance_R]))
        Spatio["Overall_Swing_Percentage"] = np.nanmean(np.concatenate([swing_L, swing_R]))

        # ----- Per-foot outputs -----
        Spatio["Stride_Length_Left"] = stride_len_L
        Spatio["Stride_Time_Left"] = stride_time_L
        Spatio["Step_Length_Left"] = step_len_L
        Spatio["Step_Time_Left"] = step_time_L
        Spatio["Speed_Left"] = speed_L
        Spatio["Stance_Percentages_Left"] = stance_L
        Spatio["Swing_Percentages_Left"] = swing_L

        Spatio["Stride_Length_Right"] = stride_len_R
        Spatio["Stride_Time_Right"] = stride_time_R
        Spatio["Step_Length_Right"] = step_len_R
        Spatio["Step_Time_Right"] = step_time_R
        Spatio["Speed_Right"] = speed_R
        Spatio["Stance_Percentages_Right"] = stance_R
        Spatio["Swing_Percentages_Right"] = swing_R

        return Spatio

    except Exception as e:
        print("Spatio-temporal calculation failed:", e)
        return {}
