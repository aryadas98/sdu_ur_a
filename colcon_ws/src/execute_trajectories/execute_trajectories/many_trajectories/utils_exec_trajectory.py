import os
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import yaml
from pathlib import Path
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt

import numpy as np
from typing import Callable, Tuple, Optional, List, Dict, Any
from skopt import Optimizer
from skopt.space import Real

FREQUENCY = 500  # Hz, the frequency at which the data is sampled

# def get_settling_time(rosbag_path):
#     df_js, df_wrench = read_rosbag(rosbag_path)
#     df_wrench = apply_lowpass_filter(df_wrench, cutoff_hz=10, fs=100)  # Example parameters
#     settling_time = get_last_settling_time(df_wrench, window=10, threshold=0.003, tail=FREQUENCY*3)
    


# def add_effort_diffs(df_js):
#     # Group by joint name and calculate effort differences
#     df_js['effort_diff'] = df_js.groupby('joint_name')['effort'].diff()
#     return df_js



def read_rosbag(rosbag_path):
    # Read the rosbag and extract the relevant data

    js_rows, wr_rows = [], []

    with Reader(rosbag_path) as reader:
        js_conns = [c for c in reader.connections if c.topic == '/joint_states']
        wr_conns = [c for c in reader.connections if '/wrench' in c.topic]
        all_conns = js_conns + wr_conns

        first_ts = None
        for conn, ts, raw in reader.messages(connections=all_conns):
            if first_ts is None:
                first_ts = ts
            t = (ts - first_ts) * 1e-9
            msg = deserialize_cdr(raw, conn.msgtype)

            if conn in js_conns:
                # append one row per joint
                for name, p, v, e in zip(msg.name, msg.position, msg.velocity, msg.effort):
                    js_rows.append({
                        'time_sec': t,
                        'joint_name': name,
                        'position': p,
                        'velocity': v,
                        'effort': e
                    })

            else:  # wrench
                w = msg.wrench
                wr_rows.append({
                    'time_sec': t,
                    'topic': conn.topic,
                    'force_x':  w.force.x,
                    'force_y':  w.force.y,
                    'force_z':  w.force.z,
                    'torque_x': w.torque.x,
                    'torque_y': w.torque.y,
                    'torque_z': w.torque.z
                })

    df_js = pd.DataFrame(js_rows).sort_values('time_sec').reset_index(drop=True)
    df_wrench = pd.DataFrame(wr_rows).sort_values('time_sec').reset_index(drop=True)

    return df_js, df_wrench

def get_settling_time(df_wrench,variance_threshold=0.5):
    df_wrench = low_pass(df_wrench)
    df_wrench = add_variance(df_wrench)
    try:
        #for i,axis in enumerate(['x', 'y', 'z']):
        var_col = 'force_var'
        # Get all points where variance is above the threshold.
        high_variance_indices = df_wrench.index[df_wrench[var_col] > variance_threshold]
        
        if not high_variance_indices.empty:
                # The last index in this series is the first point from the end to exceed the threshold.
            last_high_var_idx = high_variance_indices[-1]
                #unsettle_points.append(df_wrench.loc[last_high_var_idx])
        #latest_point = max(unsettle_points, key=lambda x: x['time_sec'])
            latest_point = df_wrench.loc[last_high_var_idx]
            return latest_point.time_sec
        else:
            print(f"No points found with variance above {variance_threshold}.")
            return None
    except:
        print("No settling time found, returning None")
        return None

def low_pass(df_wrench, cutoff_freq=15, filter_order =2, fs=FREQUENCY):


    # Calculate the Nyquist frequency
    nyquist_freq = 0.5 * fs#sample_rate

    # Avoid division by zero if sample_rate is 0
    if nyquist_freq > 0:
        normal_cutoff = cutoff_freq / nyquist_freq

        # Design the Butterworth filter
        b, a = butter(filter_order, normal_cutoff, btype='low', analog=False)

        # Apply the filter to each force component using filtfilt for zero phase shift
        df_wrench['force_x_filtered'] = filtfilt(b, a, df_wrench['force_x'])
        df_wrench['force_y_filtered'] = filtfilt(b, a, df_wrench['force_y'])
        df_wrench['force_z_filtered'] = filtfilt(b, a, df_wrench['force_z'])
    return df_wrench


def add_variance(df_wrench,FREQUENCY=500,window_size=20, n_mean=250):
    

    
    # Calculate mean from the last n_mean elements for each force component
    mean_x = df_wrench['force_x_filtered'].tail(n_mean).mean()
    mean_y = df_wrench['force_y_filtered'].tail(n_mean).mean()
    mean_z = df_wrench['force_z_filtered'].tail(n_mean).mean()
    
    # Calculate variance around the mean for the last n_last elements
    df_wrench["force_var"] = (((df_wrench['force_x_filtered'] - mean_x) ** 2)
    + ((df_wrench['force_y_filtered'] - mean_y) ** 2)
    + ((df_wrench['force_z_filtered'] - mean_z) ** 2)).rolling(window=window_size).mean()

    return df_wrench


def get_start_time(df_js, threshold=0.05):
    """
    Get the first point where the effort exceeds a given threshold.
    """
    start_point = df_js[df_js['effort'].abs() > threshold].iloc[0]
    return start_point
# start,end = get_start_time(df_js),get_settling_time(df_wrench)


def get_action_start_time(df_js, threshold=0.001, window_size=5):
    window_size = 6 * int(window_size) #  because we want to use all of the joints
    velocity_diff = df_js['velocity'].diff().abs()
    is_above_threshold = velocity_diff > threshold
    consecutive_count = is_above_threshold.rolling(window=window_size).sum()
    action_indices = consecutive_count[consecutive_count == window_size].index

    if not action_indices.empty:
        first_action_start_index = action_indices[0]
        first_action_end_index = first_action_start_index + window_size - 1
        action_segment = df_js.loc[first_action_start_index:first_action_end_index]
        return action_segment.iloc[0]['time_sec']
    else:
        return None


def get_action_end_time(df_js, threshold=0.001, window_size=5):
    window_size = 6 * int(window_size) #  because we want to use all of the joints
    velocity_diff = df_js['velocity'].diff().abs()
    is_below_threshold = velocity_diff > threshold
    consecutive_count = is_below_threshold.rolling(window=window_size).sum()
    action_indices = consecutive_count[consecutive_count == window_size].index

    if not action_indices.empty:
        last_action_end_index = action_indices[-1]
        last_action_start_index = last_action_end_index - window_size + 1
        action_segment = df_js.loc[last_action_start_index:last_action_end_index]
        return action_segment.iloc[-1]['time_sec']
    else:
        return None


def get_action_segment(df_js, action_start_time ,action_end_time):
    """
    Returns a DataFrame containing data only up to the action_end_time.
    If action_end_time is None, returns an empty DataFrame.
    """
    if action_end_time is not None and action_start_time is not None:
        df_js_action = df_js[(df_js['time_sec'] >= action_start_time) & (df_js['time_sec'] <= action_end_time)].copy()
        print(f"DataFrame 'df_js' trimmed to data before time {action_end_time:.4f}s.")
        return df_js_action
    else:
        print("action_end_time is not defined. Cannot trim the DataFrame.")
        return df_js # Create an empty dataframe


def fit_clamped_bspline_zero_end_va(
    t_real, y, *, degree=3, n_internal_knots=8,
    perturb=False,            # <-- toggle noise on/off
    noise_std=0.0,            # std dev of Gaussian noise (same units as y)
    random_state=None
):
    """
    Cubic, clamped B-spline fit with zero velocity/acceleration at both ends.
    - If perturb=True, adds N(0, noise_std^2) to the *middle* (free) control points
      before building the spline (endpoints' zero-derivative constraints preserved).
    - If perturb=False, identical behavior to the original function.
    Returns pos/vel/acc callables in REAL time and the effective control points used.
    """
    p = degree
    if p != 3:
        raise ValueError("This helper assumes cubic splines (degree=3).")
    if n_internal_knots < 3:
        raise ValueError("n_internal_knots must be >= 3 so there are free control points.")

    t_real = np.asarray(t_real, dtype=float)
    y = np.asarray(y, dtype=float)

    # Drop NaNs/Infs and sort by time
    m = np.isfinite(t_real) & np.isfinite(y)
    t_real = t_real[m]; y = y[m]
    order = np.argsort(t_real)
    t_real = t_real[order]; y = y[order]

    if len(t_real) < p + 3:
        raise ValueError("Not enough samples to fit a cubic spline.")

    t0 = t_real[0]
    t1 = t_real[-1]
    T = float(t1 - t0)
    if T <= 0:
        raise ValueError("Time vector must span a positive duration.")
    alpha = 1.0 / T

    # Normalize to u in [0,1]
    u_data = (t_real - t0) * alpha

    # Open/clamped knot vector on [0,1]
    knots_internal = np.linspace(0, 1, n_internal_knots + 2)[1:-1] if n_internal_knots > 0 else np.array([], float)
    t = np.r_[np.zeros(p+1), knots_internal, np.ones(p+1)]
    n_ctrl = len(t) - p - 1  # = n_internal_knots + 4 (for cubic)

    # Basis matrix at data positions
    def basis_col(j):
        coeff = np.zeros(n_ctrl); coeff[j] = 1.0
        return BSpline(t, coeff, p)(u_data)
    A = np.column_stack([basis_col(j) for j in range(n_ctrl)])

    # Tie first three and last three control points to start/end value
    start, end = y[0], y[-1]
    fixed_idx = [0, 1, 2, n_ctrl-3, n_ctrl-2, n_ctrl-1]
    free_idx  = [j for j in range(n_ctrl) if j not in fixed_idx]

    P_fixed = np.array([start, start, start, end, end, end], dtype=float)
    A_fixed = A[:, fixed_idx]
    A_free  = A[:, free_idx]
    rhs = y - A_fixed @ P_fixed

    # Least squares for free control points
    P_free, *_ = np.linalg.lstsq(A_free, rhs, rcond=None)

    # Base (clean) control points
    P = np.empty(n_ctrl, dtype=float)
    P[fixed_idx] = P_fixed
    P[free_idx]  = P_free

    # Optionally perturb only the middle (free) control points
    if perturb and noise_std > 0.0:
        rng = np.random.default_rng(random_state)
        P_eff = P.copy()
        P_eff[free_idx] += rng.normal(0.0, float(noise_std), size=len(free_idx))
    else:
        P_eff = P  # identical to original behavior

    # Build spline (using effective control points) and its derivatives in u-domain
    spline_u = BSpline(t, P_eff, p)
    s1_u = spline_u.derivative(1)
    s2_u = spline_u.derivative(2)

    # Real-time callables with chain rule
    def pos(t_query):
        tq = np.asarray(t_query, dtype=float)
        u = (tq - t0) * alpha
        return spline_u(u)

    def vel(t_query):
        tq = np.asarray(t_query, dtype=float)
        u = (tq - t0) * alpha
        return alpha * s1_u(u)

    def acc(t_query):
        tq = np.asarray(t_query, dtype=float)
        u = (tq - t0) * alpha
        return (alpha**2) * s2_u(u)

    return dict(
        pos=pos, vel=vel, acc=acc,
        t0=t0, T=T, alpha=alpha,
        spline_u=spline_u, s1_u=s1_u, s2_u=s2_u,
        knots=t, ctrl=P_eff  # control points actually used (perturbed or not)
    )


def add_noise_to_via_points(model, t_plot, std=0.01):
    """
    Add noise to the via points of the model for visualization.
    The start and end points remain exact.
    """
    noise = np.random.normal(0, std, t_plot.shape)
    noise[0] = 0
    noise[-1] = 0
    return model["pos"](t_plot) + noise

def run_bayes_search(
    objective: Callable[[np.ndarray], float],
    X_init: Optional[np.ndarray] = None,
    y_init: Optional[np.ndarray] = None,
    *,
    n_iter: int = 50,
    batch_size: int = 1,
    bounds: Tuple[float, float] = (0.0, 1.0),
    base_estimator: str = "GP",
    acq_func: str = "EI",
    random_state: int = 0,
    verbose: bool = True,
    #initial_traj : np.ndarray = None,
) -> Tuple[np.ndarray, float, Optimizer, List[Dict[str, Any]]]:
    """
    Bayesian optimization with warm start that logs losses as it goes.

    Returns:
      best_x, best_y, opt, history

    history = list of dicts per iteration:
      {
        'iter': int,
        'batch_X': np.ndarray (batch_size, 120),
        'batch_y': np.ndarray (batch_size,),
        'incumbent_y': float,      # best so far after this iteration
        'incumbent_x': np.ndarray  # 120-dim best so far
      }
    """
    low, high = bounds

    if X_init is not None:
        X_init = np.asarray(X_init, dtype=float)
        min_vals = np.min(X_init, axis=0)
        max_vals = np.max(X_init, axis=0)
        space = [Real(min_vals[i] + low, max_vals[i] + high, name=f"x{i}") for i in range(X_init.shape[1])]
    else:
        space = [Real(low, high, name=f"x{i}") for i in range(30)]
    opt = Optimizer(dimensions=space, base_estimator=base_estimator,
                    acq_func=acq_func, random_state=random_state)

    # Warm start
    if X_init is not None and y_init is not None:
        X_init = np.asarray(X_init, dtype=float)
        y_init = np.asarray(y_init, dtype=float).ravel()
        if X_init.ndim != 2 or X_init.shape[1] != 30:
            raise ValueError("X_init must have shape (n0, 120)")
        if y_init.shape[0] != X_init.shape[0]:
            raise ValueError("y_init length must match X_init rows")
        # if not (np.all(X_init >= low) and np.all(X_init <= high)):
        #     raise ValueError("Some initial points are out of bounds")
        opt.tell(list(map(list, X_init)), list(map(float, y_init)))

    history: List[Dict[str, Any]] = []

    # BO loop
    for it in range(1, n_iter + 1):
        asks: List[List[float]] = opt.ask(n_points=batch_size)
        ys = [float(objective(np.array(x))) for x in asks]
        opt.tell(asks, ys)

        # Incumbent (best so far)
        yi = np.array(opt.yi, float)
        Xi = np.array(opt.Xi, float)
        best_idx = int(np.argmin(yi))
        incumbent_y = float(yi[best_idx])
        incumbent_x = Xi[best_idx].copy()

        # Log
        if verbose:
            if batch_size == 1:
                print(f"[{it:03d}] y={ys[0]:.6f} | best={incumbent_y:.6f}")
            else:
                batch_str = ", ".join(f"{v:.6f}" for v in ys)
                print(f"[{it:03d}] batch_y=[{batch_str}] | best={incumbent_y:.6f}")

        # Save to history
        history.append({
            "iter": it,
            "batch_X": np.array(asks, float),
            "batch_y": np.array(ys, float),
            "incumbent_y": incumbent_y,
            "incumbent_x": incumbent_x,
        })

    # Final best
    best_x = history[-1]["incumbent_x"]
    best_y = history[-1]["incumbent_y"]
    return best_x, best_y, opt, history