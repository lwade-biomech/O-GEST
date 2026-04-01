# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 11:31:05 2025

@author: lw2175
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

def visualization_gest(INFO_L, INFO_R):
    """
    Python equivalent of Visualization_GEST.m
    Uses BSpline from SciPy to plot fitted curves.
    """

    plt.figure(figsize=(10, 6))

    # ===== LEFT FOOT (red) =====
    for CP, raw_data in zip(INFO_L["CPs"], INFO_L["Sections"]):

        CP = np.array(CP)  # 2 x Ncp
        knots = np.array(INFO_L["Knots"])

        # In MATLAB: spmak(knots, CP)
        # In SciPy: BSpline(t, c, k)
        k = INFO_L["Degree"]     # You MUST store spline degree in INFO_L
        spline = BSpline(knots, CP.T, k)

        # Evaluate spline
        t_min, t_max = knots[k], knots[-k-1]
        t_vals = np.linspace(t_min, t_max, 200)
        curve = spline(t_vals)  # shape (200, 2)

        plt.plot(curve[:, 0], curve[:, 1], color="red")
        plt.scatter(raw_data[:, 0], raw_data[:, 1], s=6, color="red")
        plt.plot(CP[0, :], CP[1, :], "-o", color="red")

    # ===== RIGHT FOOT (blue) =====
    for CP, raw_data in zip(INFO_R["CPs"], INFO_R["Sections"]):

        CP = np.array(CP)
        knots = np.array(INFO_R["Knots"])
        k = INFO_R["Degree"]

        spline = BSpline(knots, CP.T, k)

        t_min, t_max = knots[k], knots[-k-1]
        t_vals = np.linspace(t_min, t_max, 200)
        curve = spline(t_vals)

        plt.plot(curve[:, 0], curve[:, 1], color="blue")
        plt.scatter(raw_data[:, 0], raw_data[:, 1], s=6, color="blue")
        plt.plot(CP[0, :], CP[1, :], "-o", color="blue")

    # ===== Interests (black) =====
    plt.plot(INFO_L["Interests"][0], INFO_L["Interests"][1], "ok", markersize=4)
    plt.plot(INFO_R["Interests"][0], INFO_R["Interests"][1], "ok", markersize=4)

    plt.xlabel("Time")
    plt.ylabel("Depth (Horizontal)")
    plt.title("O-GEST")
    plt.grid(True)
    plt.show()
