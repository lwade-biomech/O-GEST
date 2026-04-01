# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 18:34:00 2025

@author: lw2175
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks
from scipy.signal import argrelextrema
from scipy.interpolate import BSpline


def data_checker(JL, JR, Time):
    JL = np.asarray(JL, dtype=float)
    JR = np.asarray(JR, dtype=float)
    Time = np.asarray(Time, dtype=float).flatten()

    # Combine left & right just like MATLAB
    JDL = np.hstack([JL, JR])
    n, m = JDL.shape

    # ============================================================
    # A. Trim leading and trailing NaN/zeros for each column
    # ============================================================
    lead = np.zeros(m, dtype=int)
    tail = np.zeros(m, dtype=int)

    for i in range(m):
        col = JDL[:, i]
        # Leading zeros/NaNs
        idx = np.where((col == 0) | np.isnan(col))[0]
        lead[i] = idx[-1] + 1 if len(idx) > 0 and idx[-1] != len(col)-1 else \
                  (idx[-1] + 1 if len(idx) > 0 else 0)

        # Trailing zeros/NaNs
        idx = np.where((col[::-1] == 0) | np.isnan(col[::-1]))[0]
        tail[i] = idx[-1] + 1 if len(idx) > 0 and idx[-1] != len(col)-1 else \
                  (idx[-1] + 1 if len(idx) > 0 else 0)

    trim_start = np.max(lead)
    trim_end = np.max(tail)

    if trim_start > 0:
        JDL = JDL[trim_start:, :]
        Time = Time[trim_start:]

    if trim_end > 0:
        JDL = JDL[:-trim_end, :]
        Time = Time[:-trim_end]

    # ============================================================
    # B. Fix internal gaps by linear interpolation
    # ============================================================
    for i in range(JDL.shape[1]):
        col = JDL[:, i]
        bad = (col == 0) | np.isnan(col)

        if np.any(bad):
            good_x = np.where(~bad)[0]
            bad_x = np.where(bad)[0]

            f = interp1d(good_x, col[good_x], kind="linear", fill_value="extrapolate")
            col[bad_x] = f(bad_x)

        JDL[:, i] = col

    # Split back into left/right
    m_half = JDL.shape[1] // 2
    JL = JDL[:, :m_half]
    JR = JDL[:, m_half:]

    # ============================================================
    # C. Detect shortened start
    # ============================================================
    n = JL.shape[0]
    SUB = np.abs(JL - JR)

    indic = np.zeros(m_half)
    first_peak = np.zeros(m_half)

    for i in range(m_half):
        sub = SUB[:, i]

        # MATLAB islocalmin(-SUB) = local maxima of SUB
        peaks = np.where((sub[1:-1] > sub[:-2]) & (sub[1:-1] > sub[2:]))[0] + 1
        if len(peaks) == 0:
            continue

        first_peak[i] = peaks[0]

        if (sub[:peaks[0]].min() == sub[0]) and (sub[0] < 0.1 * sub.mean()):
            indic[i] = 1

    if indic.sum() > 0:
        TH = int(np.floor(np.mean(first_peak)/10) + 1)
        JL = JL[TH:, :]
        JR = JR[TH:, :]
        Time = Time[TH:]

    # ============================================================
    # D. Detect shortened end
    # ============================================================
    indic = np.zeros(m_half)
    last_peak = np.zeros(m_half)

    SUB = np.abs(JL - JR)

    for i in range(m_half):
        sub = SUB[:, i]

        peaks = np.where((sub[1:-1] > sub[:-2]) & (sub[1:-1] > sub[2:]))[0] + 1
        if len(peaks) == 0:
            continue

        last = peaks[-1]
        last_peak[i] = last

        if (sub[last:].min() == sub[-1]) and (sub[-1] < 0.1 * sub.mean()):
            indic[i] = 1

    if indic.sum() > 0:
        TH = int(np.floor(np.mean(n - last_peak)/3) + 1)
        JL = JL[:-TH, :]
        JR = JR[:-TH, :]
        Time = Time[:-TH]

    return JL, JR, Time




def local_minima(x):
    """Equivalent of MATLAB islocalmin."""
    return argrelextrema(x, np.less)[0]

def derivative(x, t):
    """Compute abs(d(x)/d(t)) just like MATLAB."""
    dx = np.diff(x)
    dt = np.diff(t)
    return np.abs(dx / dt)

def Info_Detector(JointsDepth_L, JointsDepth_R, Time):
    """
    Python version of MATLAB Info_Detetctor.m
    """

    Time = np.asarray(Time).flatten()
    JD_L_raw = np.asarray(JointsDepth_L)[:, 0]
    JD_R_raw = np.asarray(JointsDepth_R)[:, 0]

    nL = len(JD_L_raw)
    nR = len(JD_R_raw)
    end_frame = int(np.mean([nL, nR])) - 1

    # ----------------------------------------------------------
    # STEP 1 — Smooth the signals (Savitzky–Golay filter)
    # ----------------------------------------------------------
    JD_L_filt = savgol_filter(JD_L_raw, window_length=15, polyorder=2)
    JD_R_filt = savgol_filter(JD_R_raw, window_length=15, polyorder=2)

    # ----------------------------------------------------------
    # STEP 2 — Compute difference and find local minima
    # ----------------------------------------------------------
    diff_LR = savgol_filter(JD_L_filt - JD_R_filt, 21, 2)
    Sub = np.abs(savgol_filter(diff_LR, 21, 2))

    intersections = local_minima(Sub)

    # ----------------------------------------------------------
    # STEP 3 — Filter intersections by magnitude threshold
    # ----------------------------------------------------------
    diff_at_int = JD_L_filt[intersections] - JD_R_filt[intersections]
    thresh = 0.1 * np.max(np.abs(JD_L_filt - JD_R_filt))

    good_mask = np.abs(diff_at_int) >= thresh
    intersections = intersections[good_mask]

    # ----------------------------------------------------------
    # STEP 4 — Remove intersections that do not separate peaks
    # ----------------------------------------------------------
    good2 = []
    for i in range(len(intersections) - 1):
        a = intersections[i]
        b = intersections[i + 1]
        peak_diff = np.max(Sub[a:b]) - np.min(Sub[a:b])
        if peak_diff > 0.75 * thresh:
            good2.append(intersections[i])
    if len(intersections) > 0:
        good2.append(intersections[-1])
    intersections = np.array(good2, dtype=int)

    # ----------------------------------------------------------
    # Helper: derivative filtering
    # ----------------------------------------------------------
    def derivative_threshold(D, T):
        d = derivative(D, T)
        thresh_total = np.mean(d) * 0.2
        d[d < thresh_total] = 0
        d[d < np.max(d) * 0.2] = 0
        return d

    # ----------------------------------------------------------
    # STEP 5 — Check correctness using derivative logic
    # ----------------------------------------------------------
    def is_correct_intersections(D, T, ints):
        d = derivative_threshold(D, T)
        odds = ints[1::2]
        evens = ints[0::2]

        sum_odds = np.sum(d[odds - 1])
        sum_evens = np.sum(d[evens - 1])

        return (sum_odds == 0 and sum_evens != 0) or \
               (sum_odds != 0 and sum_evens == 0)

    correct_L = is_correct_intersections(JD_L_raw, Time, intersections)
    correct_R = is_correct_intersections(JD_R_raw, Time, intersections)
    intersections_info = "Correct" if (correct_L and correct_R) else "InCorrect"

    # ----------------------------------------------------------
    # STEP 6 — Fallback: if incorrect, use derivative peaks
    # ----------------------------------------------------------
    if intersections_info == "InCorrect":
        def build_deriv_intersections(D, T):
            d = derivative_threshold(D, T)
            locmax = local_minima(-d)
            locmin = local_minima(d)
            sorted_ints = np.sort(np.concatenate([locmax, locmin]))
            return sorted_ints

        ints_L = build_deriv_intersections(JD_L_raw, Time)
        ints_R = build_deriv_intersections(JD_R_raw, Time)

        intersections = ints_L if len(ints_L) >= len(ints_R) else ints_R
        intersections_info = "Corrected_With_Thresholds"

    intersections = intersections[(intersections >= 0) & (intersections <= end_frame)]

    # ----------------------------------------------------------
    # STEP 7 — Determine whether first cycle is Left or Right
    # ----------------------------------------------------------
    if len(intersections) < 2:
        raise ValueError("Not enough intersections found.")

    mid = int((intersections[0] + intersections[1]) // 2)

    if JD_L_filt[mid] > JD_R_filt[mid]:
        start = "Left"
        idsL = np.arange(0, len(intersections), 2)
        idsR = np.arange(1, len(intersections), 2)
    else:
        start = "Right"
        idsR = np.arange(0, len(intersections), 2)
        idsL = np.arange(1, len(intersections), 2)

    # ----------------------------------------------------------
    # STEP 8 — Build cycle sections
    # ----------------------------------------------------------
    def build_sections(ids, JD, foot_name):
        if len(ids) < 2:
            return []

        sections = []
        for i in range(len(ids) - 1):
            a = intersections[ids[i]]
            b = intersections[ids[i + 1]]
            t = Time[a:b]
            d = JD[a:b, :]
            sections.append(np.column_stack([t, d]))
        return sections

    sections_L = build_sections(idsL, JointsDepth_L, "Left")
    sections_R = build_sections(idsR, JointsDepth_R, "Right")

    # ----------------------------------------------------------
    # STEP 9 — Build output dicts
    # ----------------------------------------------------------
    INFO_L = {
        "N_Cycle": max(0, len(idsL) - 1),
        "SEsamples": [intersections[idsL[0]], intersections[idsL[-1]]],
        "Time": Time[intersections[idsL[0]] : intersections[idsL[-1]]],
        "Data": JointsDepth_L[intersections[idsL[0]] : intersections[idsL[-1]]],
        "Intersections_Info": intersections_info,
        "Sections": sections_L
    }

    INFO_R = {
        "N_Cycle": max(0, len(idsR) - 1),
        "SEsamples": [intersections[idsR[0]], intersections[idsR[-1]]],
        "Time": Time[intersections[idsR[0]] : intersections[idsR[-1]]],
        "Data": JointsDepth_R[intersections[idsR[0]] : intersections[idsR[-1]]],
        "Intersections_Info": intersections_info,
        "Sections": sections_R
    }

    return INFO_L, INFO_R



def NCP_Finder(INFO):
    """
    Python translation of the MATLAB NCP_Finder function.
    INFO.Sections is assumed to be a list of Nx2 numpy arrays.
    """
    Percent = 0.015
    NCP = None
    Intensity = None

    if INFO and len(INFO["Sections"]) > 0:
        ratios = []

        for Sec in INFO["Sections"]:
            n = Sec.shape[0]

            # Baseline windows (first and last 4%)
            baseline_samples = int(np.floor(n * 0.04)) + 1
            baseline_end_idx = np.arange(n - baseline_samples, n)

            # smooth trajectory (column 2)
            Sec_D = savgol_filter(Sec[:, 1], window_length=7, polyorder=2)

            # threshold
            TH = Percent * abs(np.mean(Sec_D[:baseline_samples]) -
                               np.mean(Sec_D[baseline_end_idx]))

            # sorted values (ascending)
            SORTED = np.sort(Sec_D)
            Val = abs(np.mean(SORTED[:baseline_samples]) -
                      np.mean(Sec_D[baseline_end_idx]))

            ratio = Val / TH if TH != 0 else np.inf
            ratios.append(ratio)

        if max(ratios) >= 1:
            NCP = 5
            Intensity = "Intense"
        else:
            NCP = 4
            Intensity = "Normal"

    return NCP, Intensity



def cubic_bspline_basis_functions(Ncp):
    """
    Python translation of Cubic_Bspline_Basis_Function.
    Returns a list of SciPy BSpline objects (basis functions).
    """

    degree = 3  # cubic

    # Construct knot vector (open uniform)
    start = [0] * (degree + 1)
    middle = list(range(1, Ncp - degree))
    end = [Ncp - degree] * (degree + 1)
    knots = np.array(start + middle + end, dtype=float)

    basis_functions = []

    for i in range(Ncp):
        # control point vector (one-hot)
        coeffs = np.zeros(Ncp)
        coeffs[i] = 1.0

        # Create BSpline basis function
        basis = BSpline(knots, coeffs, degree, extrapolate=False)
        basis_functions.append(basis)

    return basis_functions



def initial_control_points(INFO):
    Sections = INFO["Sections"]
    Ncp = INFO["Ncp"]

    # Base percentages
    Percentages = np.array([0.05, 0.35, 0.65, 0.95], dtype=float)

    # If more than 4 CPs, create evenly spaced interior percentages
    if Ncp > 4:
        interior_count = 2 + (Ncp - 4)
        Percentages = np.concatenate([
            [Percentages[0]],
            np.linspace(Percentages[1], Percentages[2], interior_count),
            [Percentages[3]]
        ])

    CPs_Cycles = []

    # --- Process each section ---
    for section in Sections:
        Time = section[:, 0]
        Depths = section[:, 1]

        # Sample positions
        samples = (Percentages * len(Time)).astype(int)
        samples = np.clip(samples, 0, len(Time)-1)

        Cps_T = Time[samples]
        Cps_D = Depths[samples].copy()

        # Force boundary controls
        Cps_D[1] = Cps_D[0]
        Cps_D[-2] = Cps_D[-1]

        # If Ncp > 4, modify interior controls
        if Ncp > 4:
            for k in range(2, Ncp-2):
                Cps_D[k] = Cps_D[-2] - 0.25 * (Cps_D[1] - Cps_D[-2])

        # Store 2×Ncp matrix
        CPs_Cycles.append(np.vstack([Cps_T, Cps_D]))

    # --- Continuity enforcement between cycles ---
    for i in range(len(CPs_Cycles) - 1):
        P0_next_D = CPs_Cycles[i+1][1, 0]  # first depth of next cycle
        CPs_Cycles[i][1, -2] = P0_next_D
        CPs_Cycles[i][1, -1] = P0_next_D

    INFO["CPs"] = CPs_Cycles
    return INFO



def id_cycles_up_down_events_indexes(INFO_L, INFO_R):
    Dim = 2

    # Extract basic info
    N_Cycle_L = INFO_L["N_Cycle"]
    N_Cycle_R = INFO_R["N_Cycle"]
    CPs_L = INFO_L["CPs"]
    CPs_R = INFO_R["CPs"]
    Ncp_L = INFO_L["Ncp"]
    Ncp_R = INFO_R["Ncp"]

    # ------------------ LEFT SIDE PROCESSING ------------------
    Indexes_Events_L = []
    Other_Bezier_ID_L = []
    Other_Bezier_tk_L = []
    ID_L = []

    for i in range(N_Cycle_L):
        C1_L = CPs_L[i]
        index = i * (Ncp_L * Dim)

        P0_L = C1_L[1, 0]          # depth at first CP
        Pn_L = C1_L[1, -1]         # depth at last CP

        INDEX = [np.nan, np.nan]
        ID = ["NO", "NO"]
        Other_ID = [np.nan, np.nan]
        Other_tk = ["NaN", "NaN"]

        # Compare to all RIGHT cycles
        for j in range(N_Cycle_R):
            Cj_R = CPs_R[j]
            P0_R = Cj_R[1, 0]
            Pn_R = Cj_R[1, -1]

            # LEFT start inside RIGHT
            if P0_R >= P0_L >= Pn_R:
                ID[0] = "YES"
                INDEX[0] = index + 1
                Other_ID[0] = j + 1
                Other_tk[0] = "Second"

            # LEFT end inside RIGHT
            if P0_R >= Pn_L >= Pn_R:
                ID[1] = "YES"
                INDEX[1] = index + (Ncp_L * Dim) - (Dim - 1)
                Other_ID[1] = j + 1
                Other_tk[1] = "First"

        ID_L.append(ID)
        Indexes_Events_L.extend(INDEX)
        Other_Bezier_ID_L.extend(Other_ID)
        Other_Bezier_tk_L.extend(Other_tk)

    IDs_L = {
        "ID": ID_L,
        "Indexes_Events": np.array(Indexes_Events_L),
        "Other_Bezier_ID": Other_Bezier_ID_L,
        "Other_Bezier_tk": Other_Bezier_tk_L,
    }

    # Flatten the ID list
    flat_ids_L = [item for sublist in ID_L for item in sublist]
    IDs_L["ids"] = flat_ids_L

    # Compute how many cycles to remove at beginning
    begin_remove = 0
    for flag in flat_ids_L:
        if flag == "YES":
            break
        begin_remove += 1
    IDs_L["interests_remove_begin"] = begin_remove * 2

    # Compute end removal count
    end_remove = 0
    for flag in reversed(flat_ids_L):
        if flag == "YES":
            break
        end_remove += 1
    IDs_L["interests_remove_end"] = end_remove * 2

    # ------------------ RIGHT SIDE PROCESSING (mirror) ------------------
    Indexes_Events_R = []
    Other_Bezier_ID_R = []
    Other_Bezier_tk_R = []
    ID_R = []

    for i in range(N_Cycle_R):
        C1_R = CPs_R[i]
        index = i * (Ncp_R * Dim)

        P0_R = C1_R[1, 0]
        Pn_R = C1_R[1, -1]

        INDEX = [np.nan, np.nan]
        ID = ["NO", "NO"]
        Other_ID = [np.nan, np.nan]
        Other_tk = ["NaN", "NaN"]

        # Compare to LEFT cycles
        for j in range(N_Cycle_L):
            Cj_L = CPs_L[j]
            P0_L = Cj_L[1, 0]
            Pn_L = Cj_L[1, -1]

            if P0_L >= P0_R >= Pn_L:
                ID[0] = "YES"
                INDEX[0] = index + 1
                Other_ID[0] = j + 1
                Other_tk[0] = "Second"

            if P0_L >= Pn_R >= Pn_L:
                ID[1] = "YES"
                INDEX[1] = index + (Ncp_R * Dim) - (Dim - 1)
                Other_ID[1] = j + 1
                Other_tk[1] = "First"

        ID_R.append(ID)
        Indexes_Events_R.extend(INDEX)
        Other_Bezier_ID_R.extend(Other_ID)
        Other_Bezier_tk_R.extend(Other_tk)

    IDs_R = {
        "ID": ID_R,
        "Indexes_Events": np.array(Indexes_Events_R),
        "Other_Bezier_ID": Other_Bezier_ID_R,
        "Other_Bezier_tk": Other_Bezier_tk_R,
    }

    flat_ids_R = [item for sublist in ID_R for item in sublist]
    IDs_R["ids"] = flat_ids_R

    begin_remove = 0
    for flag in flat_ids_R:
        if flag == "YES":
            break
        begin_remove += 1
    IDs_R["interests_remove_begin"] = begin_remove * 2

    end_remove = 0
    for flag in reversed(flat_ids_R):
        if flag == "YES":
            break
        end_remove += 1
    IDs_R["interests_remove_end"] = end_remove * 2

    return IDs_L, IDs_R


###### SPLIT THIS INTO ITS OWN SCRIPT FROM HERE
def normalization_error_calculator_one_landmark(x0, INFO_L, INFO_R):
    Dim = 2

    # Split control points
    x_L = x0[:INFO_L['N_Param']]
    x_R = x0[INFO_L['N_Param']:]

    # Update INFO structures with reshaped CPs
    INFO_L['CPs'] = solution_completion_and_CPs_update(x_L, Dim, INFO_L)
    INFO_R['CPs'] = solution_completion_and_CPs_update(x_R, Dim, INFO_R)

    # Calculate geometric errors
    Error_Geo_L = geometric_error_calculator(INFO_L)
    Error_Geo_R = geometric_error_calculator(INFO_R)

    # Store in output dictionary
    Normalization_E = {
        'Geo_L0': Error_Geo_L,
        'Geo_R0': Error_Geo_R
    }

    return Normalization_E




def solution_completion_and_CPs_update(x0, Dim, INFO):
    """
    Reshape the flat control points vector into per-cycle 2xNcp arrays.
    """
    Ncp = INFO['Ncp']
    N_Cycle = len(INFO['CPs'])

    # Ensure x0 is a column vector
    x0 = np.array(x0).reshape(-1)

    # Reshape into Dim x (Ncp * N_Cycle)
    x0_reshaped = x0.reshape(Dim, Ncp * N_Cycle)

    CPs_Cycles = []

    for i in range(N_Cycle):
        start_idx = i * Ncp
        end_idx = (i + 1) * Ncp
        xx = x0_reshaped[:, start_idx:end_idx]
        CPs_Cycles.append(xx)

    return CPs_Cycles




def geometric_error_calculator(INFO):
    """
    Computes geometric error between trajectory data and cubic B-spline approximation.
    """
    Cycles_Data = INFO['Sections']
    CPs_Cycles = INFO['CPs']
    N_Cycle = len(Cycles_Data)
    
    E_All = np.zeros(N_Cycle)
    
    for i in range(N_Cycle):
        Controls = CPs_Cycles[i]       # Dim x Ncp
        raw_data = Cycles_Data[i]      # N_samples x 2
        Data = raw_data.T               # 2 x N_samples
        
        # Split into regions outside control points
        Data_Line1 = Data[:, Data[0, :] < Controls[0, 0]]
        Data_Line2 = Data[:, Data[0, :] > Controls[0, -1]]
        
        # Data for B-spline region
        mask = (Data[0, :] >= Controls[0, 0]) & (Data[0, :] <= Controls[0, -1])
        Data_Bspline = Data[:, mask]
        
        # Errors for flat regions
        E_Line1 = np.abs(Data_Line1[1, :] - Controls[1, 0])
        E_Line2 = np.abs(Data_Line2[1, :] - Controls[1, -1])
        
        # Errors for B-spline region
        _, _, Distances = find_projection_on_cubic_bspline(Controls, Data_Bspline)
        E_Bspline = Distances
        
        # Total error for this cycle
        E_All[i] = np.sum(E_Line1) + np.sum(E_Line2) + np.sum(E_Bspline)
    
    # Total error
    Error = np.sum(E_All)
    return Error




def find_projection_on_cubic_bspline(Controls, Data_Bspline):
    """
    Projects points onto a cubic B-spline defined by Controls.
    Handles 1 or 2 Bézier segments depending on Ncp.
    """
    _, Ncp = Controls.shape

    if Ncp == 4 and Data_Bspline.size != 0:
        Tk, Projections = find_projection_on_cubic_bezier(Controls, Data_Bspline)
        Distances = np.linalg.norm(Data_Bspline - Projections, axis=0)
    
    elif Ncp == 5 and Data_Bspline.size != 0:
        Beziers_Inside = beziers_inside_bspline(Controls)
        
        Tk1, Projections1 = find_projection_on_cubic_bezier(Beziers_Inside[0], Data_Bspline)
        Tk2, Projections2 = find_projection_on_cubic_bezier(Beziers_Inside[1], Data_Bspline)
        Tk2 = Tk2 + 1  # adjust parameter for second segment
        
        Distances1 = np.linalg.norm(Data_Bspline - Projections1, axis=0)
        Distances2 = np.linalg.norm(Data_Bspline - Projections2, axis=0)
        
        Distances = np.minimum(Distances1, Distances2)
        Projections = Projections1.copy()
        Tk = Tk1.copy()
        
        mask = Distances2 < Distances1
        Projections[:, mask] = Projections2[:, mask]
        Tk[mask] = Tk2[mask]
    
    else:  # Data_Bspline is empty
        Tk = np.array([])
        Projections = np.array([[], []])
        Distances = np.array([])

    return Tk, Projections, Distances




def find_projection_on_cubic_bezier(Controls, Data_Bspline):
    """
    Orthogonally projects 2D points onto a cubic Bézier curve with 4 control points.
    If orthogonal projection fails, fallback to direct projection along x.
    """
    if Data_Bspline.size == 0:
        return np.array([]), np.empty((2,0))
    
    P0, P1, P2, P3 = Controls[:,0], Controls[:,1], Controls[:,2], Controls[:,3]
    M = Data_Bspline.shape[1]
    ROOTS = np.zeros(M)
    
    for i in range(M):
        point = Data_Bspline[:,i]
        
        # Coefficients for orthogonal projection (simplified vector computation)
        co_t5 = (3*P0 - 9*P1 + 9*P2 - 3*P3) * (P0 - 3*P1 + 3*P2 - P3)
        co_t4 = -(6*P0 - 12*P1 + 6*P2)*(P0 - 3*P1 + 3*P2 - P3) - (3*P0 - 6*P1 + 3*P2)*(3*P0 - 9*P1 + 9*P2 - 3*P3)
        co_t3 = (3*P0 - 3*P1)*(3*P0 - 9*P1 + 9*P2 - 3*P3) + (3*P0 - 6*P1 + 3*P2)*(6*P0 - 12*P1 + 6*P2) + (3*P0 - 3*P1)*(P0 - 3*P1 + 3*P2 - P3)
        co_t2 = -(3*P0 - 3*P1)*(3*P0 - 6*P1 + 3*P2) - (3*P0 - 3*P1)*(6*P0 - 12*P1 + 6*P2) - (P0 - point)*(3*P0 - 9*P1 + 9*P2 - 3*P3)
        co_t1 = (3*P0 - 3*P1)**2 + (P0 - point)*(6*P0 - 12*P1 + 6*P2)
        co_t0 = -(P0 - point)*(3*P0 - 3*P1)
        
        coeffs = np.array([np.sum(co_t5), np.sum(co_t4), np.sum(co_t3), np.sum(co_t2), np.sum(co_t1), np.sum(co_t0)])
        
        # Solve polynomial for t
        roots_all = np.roots(coeffs)
        roots_real = roots_all[np.isreal(roots_all)].real
        roots_valid = roots_real[(roots_real >= -1e-4) & (roots_real <= 1+1e-4)]
        
        if len(roots_valid) == 0:
            # Fallback: direct projection along x
            Px0, Px1, Px2, Px3 = P0[0], P1[0], P2[0], P3[0]
            T_Point = point[0]
            co = [3*Px1 - Px0 - 3*Px2 + Px3, 3*Px0 - 6*Px1 + 3*Px2, 3*Px1 - 3*Px0, Px0 - T_Point]
            roots_dir = np.roots(co)
            roots_dir = roots_dir[np.isreal(roots_dir)].real
            roots_valid = roots_dir[(roots_dir >= -1e-4) & (roots_dir <= 1+1e-4)]
            
            if len(roots_valid) == 0:
                roots_valid = np.array([0]) if P0[0] > T_Point else np.array([1])
        
        ROOTS[i] = roots_valid[0]
    
    # Compute cubic Bézier points
    t = ROOTS
    B0 = (1 - t)**3
    B1 = 3 * t * (1 - t)**2
    B2 = 3 * t**2 * (1 - t)
    B3 = t**3
    Projections = (B0*P0[:,None] + B1*P1[:,None] + B2*P2[:,None] + B3*P3[:,None])
    
    Tk = ROOTS
    return Tk, Projections




def beziers_inside_bspline(Controls):
    """
    Split a cubic B-spline with 2*Ncp control points into cubic Bézier segments.
    
    Controls: np.array of shape (2, Ncp)
    Returns: list of np.array of shape (2, 4)
    """
    Dim, Ncp = Controls.shape
    Num_Beziers = (Ncp - 4) + 1
    
    if Num_Beziers == 1:
        return [Controls]
    
    # Create open uniform knot vector with multiplicities
    knots = np.concatenate((
        np.zeros(4),
        np.arange(1, Ncp-4+1),
        np.full(4, Ncp-3)
    ))
    
    # B-spline object
    sp = BSpline(knots, Controls.T, 3)  # note: scipy expects shape (N, dim)
    
    # Refine B-spline at internal knots to get separate Bézier segments
    # Internal knots for separation
    internal_knots = np.repeat(np.arange(1, Num_Beziers), 2)
    
    # Evaluate spline at knots to get control points (approximation)
    # Note: scipy doesn't provide direct coefficient splitting like MATLAB,
    # so we approximate by evaluating at t=[0, 1/3, 2/3, 1] for each segment
    Beziers_Inside = []
    for i in range(Num_Beziers):
        t0 = i
        t_vals = t0 + np.array([0, 1/3, 2/3, 1])
        pts = sp(t_vals).T  # shape (2,4)
        Beziers_Inside.append(pts)
    
    return Beziers_Inside


###########SPLIT INTO A NEW SCRIPT FROM HERE


def quadratic_system_matrices_calculator(INFO):
    Basis = INFO["Basis"]
    CPs_Cycles = INFO["CPs"]
    Cycles_Data = INFO["Sections"]
    Ncp = INFO["Ncp"]
    Dim = 2

    N_Cycle = len(Cycles_Data)

    # allocate global matrices
    H_Final = np.zeros((N_Cycle*Dim*Ncp, N_Cycle*Dim*Ncp))
    F_Final = np.zeros((N_Cycle*Dim*Ncp, 1))

    Identity_Line = np.zeros((2,2))
    Identity_Line[1,1] = 1

    E_Total = 0

    for i in range(N_Cycle):

        Controls = CPs_Cycles[i]
        raw_data = Cycles_Data[i]
        Data = raw_data[:, :2].T  # time, distance rows

        # ----- split into Line1, Line2, B-spline region -----
        Data_Line1 = Data[:, Data[0, :] < Controls[0,0]]
        Data_Line2 = Data[:, Data[0, :] > Controls[0,-1]]

        # middle region
        Data_Bspline = Data.copy()
        if Data_Line1.shape[1] > 0:
            Data_Bspline = Data_Bspline[:, Data_Line1.shape[1]:]
        if Data_Line2.shape[1] > 0:
            Data_Bspline = Data_Bspline[:, :-Data_Line2.shape[1]]

        # initialize local matrices
        F = np.zeros((Dim*Ncp, 1))
        H = np.zeros((Dim*Ncp, Dim*Ncp))

        # ------------- Line 1 contribution ----------------
        if Data_Line1.shape[1] > 0:
            F[1] += np.sum(-2 * Data_Line1[1])
            H[0:2, 0:2] += 2 * Identity_Line * Data_Line1.shape[1]

        # ------------- Line 2 contribution ----------------
        if Data_Line2.shape[1] > 0:
            last_block = Dim*Ncp - 2
            F[last_block + 1] += np.sum(-2 * Data_Line2[1])
            H[last_block:last_block+2, last_block:last_block+2] += \
                2 * Identity_Line * Data_Line2.shape[1]

        # ------------- B-spline projection ----------------
        Tk, Projections, E_Bspline = find_projection_on_cubic_bspline(Controls, Data_Bspline)

        # accumulate error
        E_Line1 = np.abs(Data_Line1[1] - Controls[1,0]) if Data_Line1.size > 0 else 0
        E_Line2 = np.abs(Data_Line2[1] - Controls[1,-1]) if Data_Line2.size > 0 else 0
        E_Total += np.sum(E_Line1) + np.sum(E_Bspline) + np.sum(E_Line2)

        # ----- Build Ni matrix (basis evaluation) -----
        Ni_Matrix = np.zeros((len(Tk), Ncp))
        for B in range(Ncp):
            Ni_Matrix[:, B] = Basis[B](Tk)

        # ----- Build f (linear term from B-spline) -----
        for B in range(Ncp):
            multiplied = Ni_Matrix[:, B][:, None] * Data_Bspline
            F[2*B:2*B+2] += -2 * np.sum(multiplied, axis=0)[:, None]

        # ----- Build H (quadratic term from B-spline) -----
        for j in range(Ni_Matrix.shape[0]):
            Ni = Ni_Matrix[j:j+1]
            mm = 2 * (Ni.T @ Ni)     # Ncp × Ncp
            # Build block matrix for 2D structure
            h = np.zeros((Dim*Ncp, Dim*Ncp))
            # x-component rows
            h[0::2, :] = np.kron(mm, np.array([[1,0]]))
            # y-component rows
            h[1::2, :] = np.kron(mm, np.array([[0,1]]))
            H += h

        # ------ Insert into global matrices ------
        start = i * Dim*Ncp
        end   = start + Dim*Ncp
        H_Final[start:end, start:end] = H
        F_Final[start:end, 0] = F[:,0]

    return H_Final, F_Final, E_Total



def solution_completion_and_cps_update(x0, Dim, INFO):
    """
    Python translation of Solution_Completion_and_CPs_Update
    """
    Ncp = INFO["Ncp"]
    CPs_Cycles = INFO["CPs"]
    N_Cycle = len(CPs_Cycles)

    # Flatten ensure x0 is a 1D vector
    x0 = np.asarray(x0).reshape(-1)

    # reshape into Dim × (Ncp * Ncycle)
    x0_matrix = x0.reshape(Dim, Ncp * N_Cycle)

    # split into cycles
    updated_cycles = []
    for i in range(N_Cycle):
        start = i * Ncp
        end   = (i + 1) * Ncp
        CP_cycle = x0_matrix[:, start:end]   # shape (Dim, Ncp)
        updated_cycles.append(CP_cycle)

    return updated_cycles


def solution_completion_and_cps_update_qp(x0, INFO, INFO_Optimization):
    """
    Python translation of Solution_Completion_and_CPs_Update_QP
    """
    Dim = 2
    Ncp = INFO["Ncp"]
    CPs_Cycles = INFO["CPs"]
    w = INFO_Optimization["Damping_Weight"]

    N_Cycle = len(CPs_Cycles)

    # Flatten x0
    x0 = np.asarray(x0).reshape(Dim, Ncp * N_Cycle)

    # Flatten old CPs to compute step
    old_cps = np.hstack(CPs_Cycles)

    # Full step (undamped)
    Step = x0 - old_cps

    updated_cycles = []
    for i in range(N_Cycle):
        start = i * Ncp
        end   = (i + 1) * Ncp

        CP_old = CPs_Cycles[i]
        CP_new_target = x0[:, start:end]

        # Damped update
        CP_new = CP_old + w * (CP_new_target - CP_old)

        updated_cycles.append(CP_new)

    return updated_cycles, Step


def Error_calculator_OneLandmark(x, INFO_L, INFO_R, INFO_Optimization, Normalization_E):

    Dim = 2
    N_L = INFO_L["N_Param"]

    # Split left and right
    x_L = x[:N_L]
    x_R = x[N_L:]

    # Update CPs
    INFO_L["CPs"] = solution_completion_and_cps_update(x_L, Dim, INFO_L)
    INFO_R["CPs"] = solution_completion_and_cps_update(x_R, Dim, INFO_R)

    # Geometric errors
    Error_Geo_L = geometric_error_calculator(INFO_L)
    Error_Geo_R = geometric_error_calculator(INFO_R)

    # Weights
    WL = INFO_Optimization["Optimization_Weights"][0]
    WR = INFO_Optimization["Optimization_Weights"][1]

    # Normalize + combine
    Error = (
        WL * (Error_Geo_L / Normalization_E["Geo_L0"]) +
        WR * (Error_Geo_R / Normalization_E["Geo_R0"])
    )

    return Error

