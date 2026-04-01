# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 16:45:09 2025

@author: lw2175
"""

import numpy as np
from scipy.linalg import block_diag

#### SPLIT THIS INTO ITS OWN SCRIPT FROM HERE
def Constraints_Creator(INFO_L, INFO_R):
    Dim = 2

    # --- Call constraint builders for each side ---
    A_InEq_L, b_InEq_L, A_Eq_L, b_Eq_L = constraints_matrices_all_cycles(Dim, INFO_L)
    A_InEq_R, b_InEq_R, A_Eq_R, b_Eq_R = constraints_matrices_all_cycles(Dim, INFO_R)

    # --- Bounds ---
    Lb_L, Ub_L = bands_lower_upper(INFO_L)
    Lb_R, Ub_R = bands_lower_upper(INFO_R)

    # --- Block diagonal assembly ---
    A_Eq = block_diag(A_Eq_L, A_Eq_R)
    b_Eq = np.concatenate([b_Eq_L, b_Eq_R], axis=0)

    A_InEq = block_diag(A_InEq_L, A_InEq_R)
    b_InEq = np.concatenate([b_InEq_L, b_InEq_R], axis=0)

    # --- Concatenate bounds ---
    Lb = np.concatenate([Lb_L, Lb_R])
    Ub = np.concatenate([Ub_L, Ub_R])

    # --- Build initial CP vector x0 ---
    # INFO.CPs is a list of arrays with shape (2,Ncp)
    CPs_L = np.hstack(INFO_L["CPs"]).flatten(order="F")
    CPs_R = np.hstack(INFO_R["CPs"]).flatten(order="F")

    x0_L = CPs_L.reshape(-1, 1)
    x0_R = CPs_R.reshape(-1, 1)

    x0 = np.vstack([x0_L, x0_R])

    # --- Store number of parameters ---
    INFO_L["N_Param"] = x0_L.shape[0]
    INFO_R["N_Param"] = x0_R.shape[0]

    return A_Eq, b_Eq, A_InEq, b_InEq, Lb, Ub, x0, INFO_L, INFO_R



def constraints_matrices_all_cycles(Dim, INFO):
    Ncp = INFO['Ncp']
    CPs_Cycles = INFO['CPs']
    N_Cycle = len(CPs_Cycles)

    # Flatten all control points
    X0 = np.zeros(N_Cycle * Dim * Ncp)
    for i, CPs in enumerate(CPs_Cycles):
        q1 = i * Ncp * Dim
        q2 = (i + 1) * Ncp * Dim
        X0[q1:q2] = CPs.T.flatten()  # transpose to match MATLAB's reshape

    # Time inequalities
    IDS = np.arange(0, Dim * Ncp * N_Cycle, 2)  # 0-based indexing
    A_Ineq_T = np.zeros(((Ncp - 1) * N_Cycle, Ncp * Dim * N_Cycle))
    b_Ineq_T = np.zeros((A_Ineq_T.shape[0],))
    for i in range(len(IDS) - 1):
        id1 = IDS[i]
        id2 = IDS[i+1]
        A_Ineq_T[i, id1] = 1
        A_Ineq_T[i, id2] = -1
        b_Ineq_T[i] = X0[id2] - X0[id1]

    # Depth inequalities
    if Ncp == 4:
        A_Ineq_D = np.zeros((N_Cycle, Ncp * Dim * N_Cycle))
        b_Ineq_D = np.zeros(N_Cycle)
        for i in range(N_Cycle):
            First = 2 + i*2*Ncp - 1   # convert to 0-based
            Last = (i+1)*2*Ncp - 1
            A_Ineq_D[i, First] = -1
            A_Ineq_D[i, Last] = 1
            b_Ineq_D[i] = X0[First] - X0[Last]
    elif Ncp == 5:
        A_Ineq_D = np.zeros((2*N_Cycle, Ncp * Dim * N_Cycle))
        b_Ineq_D = np.zeros(2*N_Cycle)
        for i in range(N_Cycle):
            First = 2 + i*2*Ncp - 1
            Last = [((i+1)*2*Ncp - 5), (i+1)*2*Ncp - 1]
            # First inequality
            A_Ineq_D[2*i, First] = -1
            A_Ineq_D[2*i, Last[1]] = 1
            b_Ineq_D[2*i] = X0[First] - X0[Last[1]]
            # Second inequality
            A_Ineq_D[2*i+1, Last[1]] = -1
            A_Ineq_D[2*i+1, Last[0]] = 1
            b_Ineq_D[2*i+1] = X0[Last[1]] - X0[Last[0]]
    else:
        A_Ineq_D = np.zeros((0, Ncp * Dim * N_Cycle))
        b_Ineq_D = np.zeros(0)

    # Combine inequalities
    A_InEq = np.vstack([A_Ineq_T, A_Ineq_D])
    b_InEq = np.zeros_like(np.concatenate([b_Ineq_T, b_Ineq_D]))

    # Depth equalities within cycles
    A_Eq_D1 = np.zeros((2*N_Cycle, Ncp*Dim*N_Cycle))
    b_Eq_D1 = np.zeros(2*N_Cycle)
    for i in range(N_Cycle):
        First = [1, 3] + i*2*Ncp
        Last = [(i+1)*2*Ncp - 3, (i+1)*2*Ncp - 1]
        A_Eq_D1[2*i, First[0]] = 1
        A_Eq_D1[2*i, First[1]] = -1
        A_Eq_D1[2*i+1, Last[0]] = 1
        A_Eq_D1[2*i+1, Last[1]] = -1

    # Depth equalities between cycles
    if N_Cycle > 1:
        A_Eq_D2 = np.zeros((N_Cycle-1, Ncp*Dim*N_Cycle))
        b_Eq_D2 = np.zeros(N_Cycle-1)
        for i in range(N_Cycle-1):
            IDs = [ (i+1)*Ncp*Dim - 1, (i+1)*Ncp*Dim + 1 - 1 ]
            A_Eq_D2[i, IDs[0]] = 1
            A_Eq_D2[i, IDs[1]] = -1
    else:
        A_Eq_D2 = np.zeros((0, Ncp*Dim*N_Cycle))
        b_Eq_D2 = np.zeros(0)

    A_Eq = np.vstack([A_Eq_D1, A_Eq_D2])
    b_Eq = np.concatenate([b_Eq_D1, b_Eq_D2])

    return A_InEq, b_InEq, A_Eq, b_Eq



def bands_lower_upper(INFO):
    Dim = 2
    Ncp = INFO['Ncp']
    Cycles_Data = INFO['Sections']
    D = INFO['Data'][:, 1]   # Depth
    T = INFO['Time']          # Time

    # Calculate threshold for significant depth changes
    diff_all = np.abs(np.diff(D) / np.diff(T))
    th_total = np.mean(diff_all) * 0.15

    Lb_L = np.zeros(Ncp * Dim * len(Cycles_Data))
    Ub_L = np.zeros(Ncp * Dim * len(Cycles_Data))

    for i, data in enumerate(Cycles_Data):
        T_cycle = data[:, 0]
        D_cycle = data[:, 1]

        diff_cycle = np.abs(np.diff(D_cycle) / np.diff(T_cycle))
        diff_cycle[diff_cycle < th_total] = 0

        MaxPos = np.argmax(diff_cycle)

        # Split into start and end
        Start_diff = diff_cycle[:MaxPos]
        End_diff = diff_cycle[MaxPos:]
        Shorten_Start = max(1, int(np.sum(Start_diff == 0) * 0.4))
        Shorten_End = max(1, int(np.sum(End_diff == 0) * 0.4))

        T_shorten_Start = T_cycle[Shorten_Start] - T_cycle[0]
        T_shorten_End = T_cycle[Shorten_End] - T_cycle[0]

        TStart = T_cycle[0] + T_shorten_Start
        TEnd = T_cycle[-1] - T_shorten_End

        DVar = np.max(D_cycle) - np.min(D_cycle)
        DUpLim = np.max(D_cycle) + 0.15 * DVar
        DDownLim = np.min(D_cycle) - 0.15 * DVar

        # Build bounds for control points
        if Ncp == 4 and Dim == 2:
            lb = [TStart, DDownLim] * Ncp
            ub = [TEnd, DUpLim] * Ncp
        elif Ncp == 5 and Dim == 2:
            lb = [TStart, DDownLim,
                  TStart, DDownLim,
                  TStart, DDownLim - 0.8*DVar,
                  TStart, DDownLim,
                  TStart, DDownLim]
            ub = [TEnd, DUpLim] * Ncp

        # Assign to full vector
        idx_start = i * Ncp * Dim
        idx_end = (i + 1) * Ncp * Dim
        Lb_L[idx_start:idx_end] = lb
        Ub_L[idx_start:idx_end] = ub

    return Lb_L, Ub_L
