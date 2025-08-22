



import glob
import shutil
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
from scipy.interpolate import BSpline


from exec_many_trajectories import URTrajectoryExecutor

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState

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


def gen_traj_from_params(params : np.ndarray) -> np.ndarray:

    lb = np.array([[ 1.3302012,  -2.31559936, -0.02970244, -1.5302012,  -0.02970244,  0.04059513],
                    [1.05429414, -2.03969231,  0.10825109, -1.25429414,  0.10825109,  0.31650218],
                    [0.68539816, -1.67079633,  0.29269908, -0.88539816,  0.29269908,  0.68539816],
                    [0.31650218, -1.30190035,  0.47714707, -0.51650218,  0.47714707,  1.05429414],
                    [0.04059513, -1.02599329,  0.6151006,  -0.24059513,  0.6151006,   1.3302012 ]])

    ub = np.array([[ 1.5302012,  -2.11559936,  0.17029756, -1.3302012,   0.17029756,  0.24059513],
                   [ 1.25429414, -1.83969231,  0.30825109, -1.05429414,  0.30825109,  0.51650218],
                   [ 0.88539816, -1.47079633,  0.49269908, -0.68539816,  0.49269908,  0.88539816],
                   [ 0.51650218, -1.10190035,  0.67714707, -0.31650218,  0.67714707,  1.25429414],
                   [ 0.24059513, -0.82599329,  0.8151006,  -0.04059513,  0.8151006,   1.5302012 ]])

    norm_params = params.reshape((-1,6))
    params = lb + (ub-lb)*norm_params


    q0 = np.array([math.pi/2, -3*math.pi/4, 0, -math.pi/2, 0, 0 ])
    qf = np.array([ 0, -math.pi/4, math.pi/4, 0, math.pi/4, math.pi/2 ])

    coeffs = np.zeros((11,6))
    coeffs[:3,:] = q0
    coeffs[-3:] = qf

    # coeffs = orig_coeffs.copy()
    coeffs[3:-3] = params

    knots = np.array([0.,    0.,    0.,    0.,    0.125, 0.25,  0.375, 0.5,   0.625, 0.75,  0.875, 1.,
                        1.,    1.,    1.   ])

    spl = BSpline(knots*3, coeffs, 3)

    fine_n_points = 20
    fine_tt = np.linspace(0, 3, fine_n_points)
    fine_qq = spl(fine_tt)
    fine_dqq = spl(fine_tt, 1)

    # import matplotlib.pyplot as plt
    # plt.plot(fine_tt, fine_qq)
    # plt.show()

    # plt.plot(fine_tt, fine_dqq)
    # plt.show()

    # plt.plot(fine_tt, spl(fine_tt, 2))
    # plt.show()

    return fine_tt, fine_qq, fine_dqq, params

def score(x: np.ndarray) -> float:
    global robot, itr

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

    tt, pos, vel, params = gen_traj_from_params(x)

    traj = JointTrajectory()
    traj.joint_names = URTrajectoryExecutor.ROBOT_JOINTS
    traj.points = [
        JointTrajectoryPoint(
            time_from_start=Duration(seconds=tt[j]+0.01).to_msg(),
            positions=pos[j],
            velocities=vel[j],
        ) for j in range(len(tt)) ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_name = f"best_{itr:05d}_{timestamp}"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bag_path = os.path.join(script_dir, "bayesian_data", bag_name)
    bag_path = os.path.abspath(bag_path)  # normalize path

    # # Ensure directory exists
    # os.makedirs(os.path.dirname(bag_path), exist_ok=True)

    # # Remove old bag file if it exists
    # if os.path.exists(bag_path):
    #     shutil.rmtree(bag_path)

    robot.exec_and_record_one(traj, bag_path)

    df_js, df_wrench = utils_exec_trajectory.read_rosbag(bag_path)
    start_time, end_time = utils_exec_trajectory.get_action_start_time(df_js), utils_exec_trajectory.get_settling_time(df_wrench)
    settling_time = end_time - start_time

    param_file = os.path.join(bag_path, "params.yaml")
    with open(param_file, "w") as f:
        yaml.dump({"timestamp":timestamp,
                    "iteration": itr,
                    "settling_time": float(settling_time),
                    "params": params.ravel().tolist()}, f)

    # Example: minimum near 0.3 (inside [0,1] box)
    print(f"Settling time {settling_time}")

    itr += 1

    return  settling_time #float(np.sum((x - 0.3)**2)) # return settling time : float(end_time - start_time)





if __name__ == "__main__":
    global robot, itr

    lb = np.array([[ 1.3302012,  -2.31559936, -0.02970244, -1.5302012,  -0.02970244,  0.04059513],
                    [1.05429414, -2.03969231,  0.10825109, -1.25429414,  0.10825109,  0.31650218],
                    [0.68539816, -1.67079633,  0.29269908, -0.88539816,  0.29269908,  0.68539816],
                    [0.31650218, -1.30190035,  0.47714707, -0.51650218,  0.47714707,  1.05429414],
                    [0.04059513, -1.02599329,  0.6151006,  -0.24059513,  0.6151006,   1.3302012 ]])

    ub = np.array([[ 1.5302012,  -2.11559936,  0.17029756, -1.3302012,   0.17029756,  0.24059513],
                   [ 1.25429414, -1.83969231,  0.30825109, -1.05429414,  0.30825109,  0.51650218],
                   [ 0.88539816, -1.47079633,  0.49269908, -0.68539816,  0.49269908,  0.88539816],
                   [ 0.51650218, -1.10190035,  0.67714707, -0.31650218,  0.67714707,  1.25429414],
                   [ 0.24059513, -0.82599329,  0.8151006,  -0.04059513,  0.8151006,   1.5302012 ]])

    lb_f = lb.ravel()
    ub_f = ub.ravel()


    rclpy.init()
    robot = URTrajectoryExecutor()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(robot)

    # Start executor in a separate thread
    exec_thread = threading.Thread(target=executor.spin, daemon=True)
    exec_thread.start()

    itr = 0

    best_params = np.array([0.4160333725474309, 0.4747201907226213, 0.7041020180079492, 0.26712424095352133, 0.3516721311937423, 0.3387159160420123, 0.39291468574860333, 0.38438493487899195, 0.5897637184838384, 0.3759567627198136, 0.38061352988133124, 0.6334508169700559, 0.7220710802263778, 0.2882782041162802, 0.5141749317943689, 0.5283168007247883, 0.5264995280839141, 0.33689441507674933, 0.4247661634606578, 0.39137455850788216, 0.38129786860761195, 0.26857187633614166, 0.35146876510796127, 0.5815136938841071, 0.43203427131173816, 0.5801786391542392, 0.30679805590024095, 0.5635378579527358, 0.4533897756058526, 0.6452045950889197])

    score(best_params)


    # Stop executor
    executor.shutdown()
    exec_thread.join(timeout=2)

    rclpy.shutdown()
