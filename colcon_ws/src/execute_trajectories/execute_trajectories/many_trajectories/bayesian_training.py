import glob
from pathlib import Path
import utils_exec_trajectory
import yaml
import matplotlib.pyplot as plt
import numpy as n
from skopt import dump, load
import numpy as np
from typing import Callable, Tuple, Optional, List, Dict, Any
from skopt import Optimizer
from skopt.space import Real
# import rclpy
# from rclpy.node import Node
# from rclpy.duration import Duration
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from control_msgs.action import FollowJointTrajectory
# from rclpy.action import ActionClient
# from sensor_msgs.msg import JointState
# from geometry_msgs.msg import WrenchStamped
# from std_msgs.msg import Float64
# from tf2_msgs.msg import TFMessage
from datetime import datetime
from enum import Enum
from datetime import timedelta
# from rclpy.serialization import serialize_message, deserialize_message
from scipy.interpolate import CubicSpline
# from exec_many_trajectories import URTrajectoryExecutor
# from generate_many_trajectories import ManyTrajGenerator
from utils_exec_trajectory import read_rosbag, get_action_start_time, get_action_end_time, get_settling_time, run_bayes_search

import pandas as pd
import numpy as np
import subprocess
import threading
import signal
import yaml
import time
import math
import csv
import os

def score(x: np.ndarray) -> float:
    """
    Example objective: Shifted Sphere (you MUST replace this).
    x: shape (120,)
    return: scalar loss (minimize)
    """
    x = np.asarray(x, dtype=float).ravel()

    #using x get the spline points
    #execute the trajectory and save rosbag path
    #rosbag_directory = 
    #rosbag_path = 
    
    # open the rosbag to get the settling time
    # df_js, df_wrench = read_rosbag(rosbag_directory)  # Replace with actual path
    #start_time, end_time = utils_exec_trajectory.get_action_start_time(df_js), utils_exec_trajectory.get_settling_time(df_wrench)
    assert x.size == 30, "score(x) expects a 30-D vector"
    # Example: minimum near 0.3 (inside [0,1] box)
    return float(np.sum((x - 0.3)**2)) # return settling time : float(end_time - start_time)





if __name__ == "__main__":
    data_dir = Path('.')
    bag_dirs = [p.parent for p in data_dir.glob('dataset_100/run_*/*.mcap*')]
    print(f"Found {len(bag_dirs)} bag directories.")
    X_data = []
    Y_data = []
    for bag_dir in bag_dirs:
        df_js, df_wrench = utils_exec_trajectory.read_rosbag(bag_dir)
        params_path = bag_dir / 'params.yaml'
        if params_path.is_file():
            with open(params_path, 'r') as f:
                params = yaml.safe_load(f)
        else:
            print(f"params.yaml not found in {bag_dir}")
            params = {}
            continue
        X = params['params']#[1:]
        print
        print(f"Processing {bag_dir.name}...")
        try:
            start_time, action_end_time, end_time = utils_exec_trajectory.get_action_start_time(df_js), utils_exec_trajectory.get_action_end_time(df_js), utils_exec_trajectory.get_settling_time(df_wrench)
            Y = end_time - start_time
            X_data.append(X)
            Y_data.append(Y)
        except:
            continue
        #df_joint = df_js[df_js['joint_name'] == df_js['joint_name'].unique()[0]]
        # plt.plot(df_joint['time_sec'], df_joint['position'], label='Position')
        # plt.title(f"Trajectory for start time : {start_time} and action end time: {action_end_time} with settling time :{end_time-start_time}")
        # plt.show()
        print(f"Start time: {start_time}, Action end time: {action_end_time}, Settling time: {end_time}")

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    print(f"X_data shape: {X_data.shape}, Y_data shape: {Y_data.shape}")
    X_init,y_init = X_data,Y_data


    # rclpy.init()
    # executor = rclpy.executors.MultiThreadedExecutor()
    # robot = URTrajectoryExecutor()
    # executor.add_node(robot)

    # # Start executor in a separate thread
    # exec_thread = threading.Thread(target=executor.spin, daemon=True)
    # exec_thread.start()

    best_x, best_y, opt, hist = run_bayes_search(
        objective=score,            # <-- plug your own function here
        X_init=X_init,              # <-- your collected inputs (n0,120)
        y_init=y_init,              # <-- their evaluated scores (n0,)
        n_iter=2,                  # additional BO iterations
        batch_size=1,               # suggest 4 points per iteration
        bounds=(-3/2*np.pi, 3/2*np.pi),          # change if your variables have different ranges
        base_estimator="ET",        # "ET" (extra-trees) handles 120D well; try "RF" too
        acq_func="EI",
        random_state=0
    )
    dump(opt, "opt_state.pkl", compress=3)

