# a python code to extract key features from the envelop
# peak force
# displacement when force is at peak
# secant stiffness of the wall when force is 15% of the peak force in the prepeak phase
# secant stiffness of the wall when displacement is 0.2

import numpy as np
import pandas as pd


def _interp_y_at_x(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    """Linear interpolation of y(x) at xq. x must be increasing."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if xq < x.min() or xq > x.max():
        raise ValueError(f"xq={xq} is outside x range [{x.min()}, {x.max()}].")
    return float(np.interp(xq, x, y))


def _first_x_where_y_reaches(x: np.ndarray, y: np.ndarray, yq: float) -> float:
    """
    Return the FIRST x where y reaches yq (y crosses from below to >= yq),
    using linear interpolation between the bracketing points.
    Assumes x is increasing.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    idx = np.where(y >= yq)[0]
    if len(idx) == 0:
        raise ValueError(f"Target y={yq} is never reached (max y={y.max()}).")

    i = int(idx[0])
    if i == 0:
        return float(x[0])

    x0, x1 = x[i - 1], x[i]
    y0, y1 = y[i - 1], y[i]

    if y1 == y0:
        return float(x1)  # flat segment

    return float(x0 + (yq - y0) * (x1 - x0) / (y1 - y0))


def extract_features_from_envelope(csv_path: str,
                                  disp_col="top_displacement",
                                  force_col="horizontal_force",
                                  disp_for_secant=0.2,
                                  frac_force_for_secant=0.15) -> dict:
    """
    Extract key features from an already-computed envelope curve (no envelope-making here).

    Features:
    - peak_force
    - displacement_at_peak
    - secant stiffness when force = 15% of peak force (pre-peak only): K = F / d_at(F)
    - secant stiffness at displacement = 0.2: K = F(0.2) / 0.2

    Notes:
    - Data is sorted by displacement before analysis.
    - If duplicate displacements exist, the last one after sorting is used.
    """
    df = pd.read_csv(csv_path)[[disp_col, force_col]].dropna().copy()

    # Ensure increasing displacement
    if  df[disp_col].iloc[1] < 0 and df[disp_col].iloc[2] < 0:# negative load
        df = df.sort_values(disp_col,ascending = False, kind="mergesort")  # stable sort
        disp = -1*df[disp_col].to_numpy(dtype=float)
        force = -1*df[force_col].to_numpy(dtype=float)
    elif df[disp_col].iloc[1] > 0 and df[disp_col].iloc[2] > 0:# positiive load
        df = df.sort_values(disp_col, kind="mergesort")  # stable sort
        disp = df[disp_col].to_numpy(dtype=float)
        force = df[force_col].to_numpy(dtype=float)

    # Peak force and displacement at peak (first occurrence)
    peak_idx = int(np.argmax(force))
    peak_force = float(force[peak_idx])
    disp_at_peak = float(disp[peak_idx])

    # Pre-peak arrays (inclusive of peak point)
    disp_pre = disp[:peak_idx + 1]
    force_pre = force[:peak_idx + 1]

    # Secant stiffness at 15% of peak (pre-peak)
    target_force = frac_force_for_secant * peak_force
    d_at_target = _first_x_where_y_reaches(disp_pre, force_pre, target_force)
    k_secant_15pct = np.inf if d_at_target == 0 else float(target_force / d_at_target)

    # Secant stiffness at displacement = 0.2
    f_at_0p2 = _interp_y_at_x(disp, force, disp_for_secant)
    k_secant_0p2 = np.inf if disp_for_secant == 0 else float(f_at_0p2 / disp_for_secant)

    return {
        "peak_force": peak_force,
        "displacement_at_peak": disp_at_peak,
        "target_force_15pct_peak": float(target_force),
        "displacement_at_15pct_peak_force_prepeak": float(d_at_target),
        "secant_stiffness_at_15pct_peak_force_prepeak": k_secant_15pct,
        "force_at_displacement_0p2": float(f_at_0p2),
        "secant_stiffness_at_displacement_0p2": k_secant_0p2,
    }


if __name__ == "__main__":
    # Example:
    # 1) Save your envelope CSV as "envelope.csv"
    # 2) Run: python this_script.py
    out = extract_features_from_envelope("../data_test/SS_02/envelop_SW2_pos.csv")
    for k, v in out.items():
        print(f"{k}: {v}")
