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
        unsettle_points = []
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
            return latest_point
        else:
            print(f"No points found with variance above {variance_threshold}.")
            return None
    except:
        print("No settling time found, returning None")
        return None
    return latest_point

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


def add_variance(df_wrench,FREQUENCY=500,window_size=100, n_mean=250):
    

    
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
