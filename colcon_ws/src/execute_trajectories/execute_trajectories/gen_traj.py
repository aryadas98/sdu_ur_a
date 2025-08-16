import csv
import os
import math
import numpy as np

# ROBOT_JOINTS = [
#     'shoulder_pan_joint',
#     'shoulder_lift_joint',
#     'elbow_joint',
#     'wrist_1_joint',
#     'wrist_2_joint',
#     'wrist_3_joint'
# ]

def default_traj():
    HOME = np.array([[0, -np.pi/2, 0, -np.pi/2, 0, 0]])

    pos = np.vstack([HOME-np.pi/4, HOME+np.pi/4])
    vel = np.zeros((2,6))
    tt = np.array([5.0, 10.0])

    traj = {"positions": pos, "velocities": vel, "times": tt}

    return traj

def my_traj():
    HOME = np.array([0, math.radians(-111.26), math.radians(112.08),
                math.radians(269.33), math.radians(-89.87), math.radians(95.92)])
    
    HOME_VEL = np.zeros((6,))

    tf = 10  # seconds
    N = 10  # no of points

    A = math.pi/4
    B = -math.pi/4

    tt = np.linspace(0, tf, N)
    base_ang = (A/2) * (1-np.cos(2*np.pi*tt/tf)) + B
    base_ang_vel = A * (np.pi/tf) * np.sin(2*np.pi*tt/tf)

    pos = np.hstack((base_ang.reshape((N,1)), np.repeat(HOME[1:].reshape((1,-1)), N, 0)))
    vel = np.hstack((base_ang_vel.reshape((N,1)), np.repeat(HOME_VEL[1:].reshape((1,-1)), N, 0)))

    traj = {"positions": pos, "velocities": vel, "times": tt}

    return traj

def write_traj(file_path, traj):
     with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)

        n_joints = traj["positions"].shape[1]

        # header
        header = ["time"] + [f"pos_{i+1}" for i in range(n_joints)] + [f"vel_{i+1}" for i in range(n_joints)]
        writer.writerow(header)

        # write rows
        for i in range(len(traj["times"])):
            row = [traj["times"][i]] + traj["positions"][i].tolist() + traj["velocities"][i].tolist()
            writer.writerow(row)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "traj.csv")
    
    # traj = default_traj()
    traj = my_traj()
    write_traj(file_path, traj)
