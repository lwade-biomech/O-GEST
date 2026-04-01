# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 11:30:11 2025

@author: lw2175
"""

# gest_one_landmark.py
import numpy as np
from scipy.optimize import minimize
import cvxpy as cp  # cvxpy required for QP path; can replace with osqp/cvxopt if desired (USEFUL FOR GPU BATCHED SOLVES - MAY HELP FOR SPEED)

# The helper names are identical to MATLAB counterparts for traceability.


def gest_one_landmark(time, joints_L, joints_R, setting):
    """
    Port of MATLAB GEST_OneLandmark to Python.

    Parameters
    ----------
    time : (N,) array
    joints_L : (N,) array  -- left landmark (single column)
    joints_R : (N,) array  -- right landmark (single column)
    setting : dict-like (will be read / partially updated)

    Returns
    -------
    INFO_L1, INFO_R1 : dict-like objects containing fields used by downstream code.
    """

    # ---------- 1) Data checking / cleaning ----------
    joints_L, joints_R, time = Data_Checker(joints_L, joints_R, time)

    # ---------- 2) Info detection (estimates N_cycle, initial params, etc.) ----------
    INFO_L1, INFO_R1 = Info_Detector(joints_L, joints_R, time)

    # ---------- 3) Determine NCP and Intensity for each side ----------
    setting["NCP_L1"], setting["Intensity_L1"] = NCP_Finder(INFO_L1)
    setting["NCP_R1"], setting["Intensity_R1"] = NCP_Finder(INFO_R1)

    INFO_L1["Intensity"] = setting["Intensity_L1"]
    INFO_R1["Intensity"] = setting["Intensity_R1"]

    INFO_L1["Ncp"] = setting["NCP_L1"]
    INFO_R1["Ncp"] = setting["NCP_R1"]

    # ---------- 4) Choose optimizer (SQP if any Ncp==5 otherwise QP) ----------
    if INFO_L1["Ncp"] == 5 or INFO_R1["Ncp"] == 5:
        optimizer = "SQP"
        setting["Optimizer"] = optimizer
    else:
        optimizer = "QP"
        setting["Optimizer"] = optimizer

    # ---------- 5) Build cubic B-spline basis & knots ----------
    INFO_L1["Basis"] = Cubic_Bspline_Basis_Function(INFO_L1["Ncp"])
    INFO_R1["Basis"] = Cubic_Bspline_Basis_Function(INFO_R1["Ncp"])
    INFO_L1["Knots"] = np.concatenate((np.zeros(4), np.arange(1, INFO_L1["Ncp"] - 3), np.ones(4)*(INFO_L1["Ncp"]-3)))
    INFO_R1["Knots"] = np.concatenate((np.zeros(4), np.arange(1, INFO_R1["Ncp"] - 3), np.ones(4)*(INFO_R1["Ncp"]-3)))

    # ---------- 6) Initial control points ----------
    INFO_L1 = Initial_Control_Points(INFO_L1)
    INFO_R1 = Initial_Control_Points(INFO_R1)

    # ---------- 7) ID cycles & create constraints ----------
    INFO_L1["IDs"], INFO_R1["IDs"] = ID_Cycles_Up_Down_Events_Indexes(INFO_L1, INFO_R1)
    (A_Eq, b_Eq, A_InEq, b_InEq,
     Lb, Ub, x0, INFO_L1, INFO_R1) = Constraints_Creator(INFO_L1, INFO_R1)

    INFO_Optimization = {}
    INFO_Optimization["Optimization_Weights"] = np.array([1.0, 1.0])

    # ---------- 8) Normalization error for initial x0 ----------
    Normalization_E = Normalization_Error_calculator_OneLandmark(x0, INFO_L1, INFO_R1)

    # ---------- 9) MaxIteration selection based on cycles ----------
    Total_Cycles = INFO_L1["N_Cycle"] + INFO_R1["N_Cycle"]
    if Total_Cycles <= 5:
        MaxIteration = 500
    elif Total_Cycles <= 10:
        MaxIteration = 750
    elif Total_Cycles <= 15:
        MaxIteration = 1000
    else:
        MaxIteration = 1250

    # ---------- 10) Intersections finder (pre-optimization) ----------
    INFO_L1, INFO_R1 = Intersections_Finder(INFO_L1, INFO_R1)

    # ================== Optimization ===================
    if optimizer == "SQP":
        print("Sequential Quadratic Programming (SQP) Is Running ...")

        # define objective for minimize (wrap Error_calculator_OneLandmark)
        def obj_fun(x):
            return Error_calculator_OneLandmark(x, INFO_L1, INFO_R1, INFO_Optimization, Normalization_E)

        # constraints for SLSQP: equality and inequality
        cons = []
        if A_Eq is not None:
            # A_Eq x = b_Eq  -> in scipy as dict type
            def eq_factory(A, b, idx):
                def eqc(x):
                    return A[idx].dot(x) - b[idx]
                return eqc
            # build equality constraints row-wise
            for irow in range(A_Eq.shape[0]):
                cons.append({'type': 'eq', 'fun': eq_factory(A_Eq, b_Eq, irow)})

        if A_InEq is not None:
            # A_InEq x <= b_InEq
            def ineq_factory(A, b, idx):
                def ineqc(x):
                    return b[idx] - A[idx].dot(x)
                return ineqc
            for irow in range(A_InEq.shape[0]):
                cons.append({'type': 'ineq', 'fun': ineq_factory(A_InEq, b_InEq, irow)})

        # bounds
        bounds = None
        if Lb is not None and Ub is not None:
            bounds = [(Lb[i], Ub[i]) for i in range(len(Lb))]

        res = minimize(obj_fun, x0.flatten(), method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 7000, 'ftol': 1e-6})
        x = res.x
        Dim = 2
        x_L = x[:INFO_L1["N_Param"]]
        x_R = x[INFO_L1["N_Param"]:]
        INFO_L1["CPs"] = Solution_Completion_and_CPs_Update(x_L, Dim, INFO_L1)
        INFO_R1["CPs"] = Solution_Completion_and_CPs_Update(x_R, Dim, INFO_R1)
        print("SQP Optimization Finished")

    elif optimizer == "QP":
        INFO_Optimization["MaxIteration"] = MaxIteration
        INFO_Optimization["Epsilon"] = 2e-4
        INFO_Optimization["Damping_Weight"] = 0.5

        Iter = 1
        WL = INFO_Optimization["Optimization_Weights"][0]
        WR = INFO_Optimization["Optimization_Weights"][1]

        NORM = np.zeros(INFO_Optimization["MaxIteration"])
        ERROR = np.zeros(INFO_Optimization["MaxIteration"])
        X_All = np.zeros((INFO_Optimization["MaxIteration"], x0.size))

        Step_Norm = 1.0
        print("Quadratic Programming (QP) Is Running ...")
        print(f"Maximum Iterations: {MaxIteration}")
        print("Iteration :  ", end="", flush=True)

        while Step_Norm >= INFO_Optimization["Epsilon"] and Iter < INFO_Optimization["MaxIteration"]:
            X_All[Iter-1, :] = x0.flatten()

            # compute quadratic system matrices for left and right
            H_L, F_L, E_Geo_L = Quadratic_System_Matrixes_Calculator(INFO_L1)
            H_R, F_R, E_Geo_R = Quadratic_System_Matrixes_Calculator(INFO_R1)

            # Normalizing and Weight applying
            H_L = WL * (H_L / Normalization_E["Geo_L0"])
            F_L = WL * (F_L / Normalization_E["Geo_L0"])
            H_R = WR * (H_R / Normalization_E["Geo_R0"])
            F_R = WR * (F_R / Normalization_E["Geo_R0"])

            # Combine
            H_Combined = block_diag(H_L, H_R)
            F_Combined = np.concatenate((F_L, F_R), axis=0)

            # Solve Quadratic Program: minimize (1/2) x'Hx + f'x subject to bounds and linear constraints
            # We'll solve using cvxpy for clarity. Replace with osqp if you prefer.
            n_vars = H_Combined.shape[0]
            x_var = cp.Variable(n_vars)
            objective = 0.5 * cp.quad_form(x_var, H_Combined) + F_Combined.flatten().T @ x_var

            constraints = []
            if A_InEq is not None:
                constraints.append(A_InEq @ x_var <= b_InEq)
            if A_Eq is not None:
                constraints.append(A_Eq @ x_var == b_Eq)
            if Lb is not None:
                constraints.append(x_var >= Lb)
            if Ub is not None:
                constraints.append(x_var <= Ub)

            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver=cp.OSQP, verbose=False)  # OSQP or default solver

            x0 = x_var.value.reshape((-1, 1))

            # Update control points and compute step sizes
            x0_L = x0[:INFO_L1["N_Param"], 0]
            x0_R = x0[INFO_L1["N_Param"]:, 0]

            INFO_L1["CPs"], Step_Left = Solution_Completion_and_CPs_Update_QP(x0_L, INFO_L1, INFO_Optimization)
            INFO_R1["CPs"], Step_Right = Solution_Completion_and_CPs_Update_QP(x0_R, INFO_R1, INFO_Optimization)

            Step_Norm = np.linalg.norm(np.array([Step_Left, Step_Right]))
            NORM[Iter-1] = Step_Norm
            ERROR[Iter-1] = WL * (E_Geo_L / Normalization_E["Geo_L0"]) + WR * (E_Geo_R / Normalization_E["Geo_R0"])

            Iter += 1
            print(f"{Iter-1}", end="", flush=True)

        print("\nStopping Criteria Satisfied")
        print("QP Is Finished")

        # Select best iterate based on minimum ERROR
        LOC = np.argmin(ERROR[:Iter-1])
        if LOC != (Iter-2):
            x0 = X_All[LOC, :].reshape((-1, 1))
            x0_L = x0[:INFO_L1["N_Param"], 0]
            x0_R = x0[INFO_L1["N_Param"]:, 0]
            INFO_L1["CPs"], _ = Solution_Completion_and_CPs_Update_QP(x0_L, INFO_L1, INFO_Optimization)
            INFO_R1["CPs"], _ = Solution_Completion_and_CPs_Update_QP(x0_R, INFO_R1, INFO_Optimization)

    # ---------- Final intersection/extremity updates ----------
    INFO_L1, INFO_R1 = Intersections_Finder(INFO_L1, INFO_R1)
    INFO_L1, INFO_R1 = Extremities_Finder_Varying(INFO_L1, INFO_R1)

    return INFO_L1, INFO_R1


# ----------------- Helper utilities & placeholders -----------------

def block_diag(A, B):
    """Simple block diagonal concatenation for two square arrays."""
    r1, c1 = A.shape
    r2, c2 = B.shape
    out = np.zeros((r1 + r2, c1 + c2))
    out[:r1, :c1] = A
    out[r1:, c1:] = B
    return out

# --- TODO: Implement the following helper functions to match MATLAB behavior ---
def Data_Checker(JL, JR, Time):
    # Validate shapes, interpolate missing frames, reshape to (N,) arrays
    # For now assume proper shape
    return JL.flatten(), JR.flatten(), Time.flatten()

def Info_Detector(JL, JR, Time):
    # Analyze signals to find N_Cycle, N_Param, initial CP guesses, etc.
    # Return two dicts with required keys (Ncp, N_Cycle, N_Param, CPs, ...)
    # Placeholder implementation (needs real port)
    INFO_L = {
        "Ncp": 8,
        "N_Cycle": 5,
        "N_Param": 16,  # example
        "CPs": None
    }
    INFO_R = {
        "Ncp": 8,
        "N_Cycle": 5,
        "N_Param": 16,
        "CPs": None
    }
    return INFO_L, INFO_R

def NCP_Finder(INFO):
    # Decide number of control points and intensity metric
    # Placeholder: return INFO['Ncp'] and intensity 1.0
    return INFO.get("Ncp", 8), 1.0

def Cubic_Bspline_Basis_Function(ncp):
    # Compute cubic B-spline basis function representation (placeholder)
    return {"ncp": ncp}

def Initial_Control_Points(INFO):
    # Make initial guess for control points. Placeholder: zeros
    INFO["CPs"] = np.zeros((INFO.get("N_Param", 16)//2, 2))  # (n_ctrl_pts, 2D)
    return INFO

def ID_Cycles_Up_Down_Events_Indexes(INFO_L, INFO_R):
    # Identify cycle indexes for up/down phases. Placeholder
    return None, None

def Constraints_Creator(INFO_L, INFO_R):
    # Build A_Eq, b_Eq, A_InEq, b_InEq, Lb, Ub, x0, etc.
    # Placeholder very simple example that sets bounds and x0
    NpL = INFO_L.get("N_Param", 16)
    NpR = INFO_R.get("N_Param", 16)
    Np = NpL + NpR
    A_Eq = None
    b_Eq = None
    A_InEq = None
    b_InEq = None
    Lb = -np.ones((Np, 1)) * 1e6
    Ub = np.ones((Np, 1)) * 1e6
    x0 = np.zeros((Np, 1))
    return A_Eq, b_Eq, A_InEq, b_InEq, Lb, Ub, x0, INFO_L, INFO_R

def Normalization_Error_calculator_OneLandmark(x0, INFO_L, INFO_R):
    # Compute normalization constants Geo_L0, Geo_R0, etc.
    return {"Geo_L0": 1.0, "Geo_R0": 1.0}

def Intersections_Finder(INFO_L, INFO_R):
    # refine intersections (placeholder)
    return INFO_L, INFO_R

def Quadratic_System_Matrixes_Calculator(INFO):
    # Return H (Hessian), F (linear term), and E_Geo (geo error)
    # Placeholder: small identity Hessian and zeros linear term
    nparam = INFO.get("N_Param", 16)//2
    H = np.eye(nparam)
    F = np.zeros((nparam, 1))
    E_geo = 0.0
    return H, F, E_geo

def Solution_Completion_and_CPs_Update(x_L, Dim, INFO):
    # Convert solution vector to CP matrix (Dim=2 typically)
    # Placeholder: reshape
    cp_count = len(x_L) // Dim
    CPs = np.reshape(x_L, (cp_count, Dim))
    return CPs

def Solution_Completion_and_CPs_Update_QP(x_L, INFO, INFO_Optimization):
    # Similar to above but return also step norm (magnitude of change)
    CPs = Solution_Completion_and_CPs_Update(x_L, 2, INFO)
    # Placeholder step size: norm of x_L
    step = np.linalg.norm(x_L)
    return CPs, step

def Error_calculator_OneLandmark(x, INFO_L, INFO_R, INFO_Optimization, Normalization_E):
    # Compute objective scalar given concatenated parameter vector x
    # Placeholder: quadratic objective using identity Hessians
    return float(np.dot(x, x))
    
def Extremities_Finder_Varying(INFO_L, INFO_R):
    # Update extremities (placeholder)
    return INFO_L, INFO_R
