import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
# from geometry_msgs.msg import WrenchStamped
# from std_msgs.msg import Float64
# from tf2_msgs.msg import TFMessage
from datetime import datetime
from enum import Enum
from datetime import timedelta
# from rclpy.serialization import serialize_message, deserialize_message
from scipy.interpolate import CubicSpline

from generate_many_trajectories import ManyTrajGenerator

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


class URTrajectoryExecutor(Node):
    ROBOT_JOINTS = [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]

    class TrajectoryError(Enum):
        SUCCESS = 0
        GOAL_REJECTED = 1
        TIMEOUT = 2
        COMMUNICATION_ERROR = 3
        UNKNOWN_ERROR = 99

    def __init__(self):
        super().__init__('ur_trajectory_executor')
        # Can be changed to whatever motion controller you want. Just be aware of what action definition the controller uses.
        # passthrough_trajectory_controller uses the same action definition, and might lighten the load on your computer, but does not publish feedback.
        self.trajectory_client = ActionClient(
            self, FollowJointTrajectory, '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        self.trajectory_client.wait_for_server()
        # self.lock = threading.Lock()


    # # Both topics are published at 500 Hz, but rclpy is not necessarily able to keep up with that rate.
    # # store the current joint state
    # def joint_states_callback(self, msg: JointState):
    #     name_to_index = {name: i for i, name in enumerate(msg.name)}
    #     positions_ordered = np.array([msg.position[name_to_index[j]] for j in self.ROBOT_JOINTS])
        
    #     with self.lock:
    #         self.curr_joint_states = positions_ordered

    def send_trajectory(self, trajectory: JointTrajectory, timeout_sec=10.0):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance = Duration(seconds=1).to_msg()
        
        try:
            # Send goal asynchronously
            future = self.trajectory_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)

            goal_handle = future.result()
            if not goal_handle.accepted:
                print("Goal was rejected by the action server")
                return self.TrajectoryError.REJECTED, None

            print("Trajectory goal accepted, waiting for result...")
            result_future = goal_handle.get_result_async()

            rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)

            result = result_future.result()
            if result is None:
                print("Trajectory result unavailable")
                return self.TrajectoryError.COMMUNICATION_ERROR, None

            print("Trajectory execution finished successfully")
            return self.TrajectoryError.SUCCESS, result

        except Exception as e:
            print(f"Unexpected error while sending trajectory: {e}")
            return self.TrajectoryError.UNKNOWN_ERROR, None


    def move_to_point(self, target_point, nsec : float):
        # move the robot to a target point in nsecs
        
        # # get the current position
        # self.curr_joint_states = None

        # curr_joint_states_sub = self.create_subscription(
        #     JointState, '/joint_states', self.joint_states_callback, 10
        # )

        # # Wait until first message arrives
        # while self.curr_joint_states is None:
        #     self.get_clock().sleep_for(Duration(seconds=0.1))

        # with self.lock:
        #     # Unsubscribe
        #     self.destroy_subscription(curr_joint_states_sub)
        #     curr_joint_states_sub = None

        # # create cubic spline trajectory from current to start point
        # q0 = current_point
        # qf = target_point
        # T = nsec
        # n_points = 10

        # tt = np.linspace(0, T, n_points)

        # times = np.array([0, T])
        # positions = np.vstack((q0, qf))

        # # Cubic spline with zero velocity at start & end
        # cs = CubicSpline(times, positions, bc_type='clamped', axis=0)

        # qq = cs(tt)
        # dqq = cs(tt, 1)

        traj = JointTrajectory()
        traj.joint_names = self.ROBOT_JOINTS
        # traj.points = [
        #     JointTrajectoryPoint(positions = qq[i],
        #         velocities = dqq[i],
        #         time_from_start = Duration(seconds=tt[i]).to_msg())
        #         for i in range(n_points)                
        # ]  # allow nsecs to go to home

        traj.points = [
            JointTrajectoryPoint(positions = target_point,
                                 velocities = np.zeros((6,), dtype=float),
                    time_from_start = Duration(seconds=nsec).to_msg())
        ]  # allow nsecs to go to home

        return self.send_trajectory(traj)

    
    def exec_and_record_one(self, traj : JointTrajectory, rosbag_path):
        
        rosbag_proc = None
            
        try:
            # start rosbag recording
            
            rosbag_cmd = [
                    "ros2", "bag", "record",
                    "/joint_states",
                    "/scaled_joint_trajectory_controller/follow_joint_trajectory/feedback",
                    "/force_torque_sensor_broadcaster/wrench",
                    "/tf",
                    "/tf_static",
                    "/speed_scaling_state_broadcaster/speed_scaling",
                    "-o", rosbag_path,
                    # "--compression-mode", "file",
                    # "--compression-format", "zstd"
                ]


            rosbag_proc = subprocess.Popen(
                    rosbag_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid  # so we can kill the whole process group
                )
            

            # now check whether the robot is close to the start point
            start_point = traj.points[0].positions

            err, res = self.move_to_point(start_point, 3)  # give 3 seconds to move to start


            if not err == self.TrajectoryError.SUCCESS:
                return err, res

            # record for 4s before maneuver
            self.get_clock().sleep_for(Duration(seconds=1))

            # Step 4: Execute trajectory
            print("Executing trajectory...")
            err, res = self.send_trajectory(traj)

            # record for 3s after maneuver
            self.get_clock().sleep_for(Duration(seconds=4))

        finally:
            # Step 5: Always stop recording, even if error occurs
            if rosbag_proc is not None:
                print("Stopping recording...")
                os.killpg(os.getpgid(rosbag_proc.pid), signal.SIGINT)
                rosbag_proc.wait()
            
            

        # Step 6: Return explicit codes
        if not err == self.TrajectoryError.SUCCESS:
            print(f"Trajectory failed: {err}")
        else:
            print("Trajectory executed successfully.")

        return err, res
    

    def exec_and_record_many(self, traj_gen: ManyTrajGenerator, N, start_idx = 0):
        start_time = time.time()
        
        # execute and record N varied trajectories
        for i in range(start_idx, N):
            ### progress info
            elapsed = time.time() - start_time
            avg_time = elapsed / (i - start_idx + 1)
            remaining = avg_time * (N - i - 1)

            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remaining_str = str(timedelta(seconds=int(remaining)))

            print(f"\n[{i+1}/{N}] Elapsed: {elapsed_str}, ETA: {remaining_str}")

            ### main loop
            while True:  # loop for retrying
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                bag_name = f"run_{i:05d}_{timestamp}"

                script_dir = os.path.dirname(os.path.abspath(__file__))
                bag_path = os.path.join(script_dir, "..", "..", "..", "..", "data", "dataset", bag_name)
                bag_path = os.path.abspath(bag_path)  # normalize path

                _traj = traj_gen.get_i_traj(i)

                traj = JointTrajectory()
                traj.joint_names = URTrajectoryExecutor.ROBOT_JOINTS
                traj.points = [
                    JointTrajectoryPoint(
                        time_from_start=Duration(seconds=_traj[0][j]+0.01).to_msg(),
                        positions=_traj[1][j],
                        velocities=_traj[2][j],
                    ) for j in range(len(_traj[0])) ]
                
                
                err, res = self.exec_and_record_one(traj, bag_path)
                
                if err == self.TrajectoryError.SUCCESS:
                    # Save parameters to YAML inside the bag folder
                    params = traj_gen.get_i_params(i)
                    param_file = os.path.join(bag_path, "params.yaml")
                    with open(param_file, "w") as f:
                        yaml.dump({"timestamp":timestamp,
                                   "iteration": i,
                                   "total_itr": N,
                                   "params": params.tolist()}, f)


                    break  # success → move to next trajectory

                # otherwise → failure handling
                print(f"❌ Error executing trajectory {i}")
                choice = input("Retry [r], Skip [s], or Stop [q]? ").lower().strip()

                if choice == "r":
                    print("Retrying...")
                    continue
                elif choice == "s":
                    print("Skipping...")
                    break
                elif choice == "q":
                    print("Stopping execution loop.")
                    return
                else:
                    return


        


def main():
    N = 10
    l_bound = np.array([-0.2, math.pi/2-0.2, 3-0.2])
    u_bound = np.array([0.2, math.pi/2+0.2, 3+0.2])

    traj_gen = ManyTrajGenerator(n_traj=N, n_params=3, l_bound=l_bound, u_bound=u_bound, seed=42)

    rclpy.init()
    executor = rclpy.executors.MultiThreadedExecutor()
    robot = URTrajectoryExecutor()
    executor.add_node(robot)

    # Start executor in a separate thread
    exec_thread = threading.Thread(target=executor.spin, daemon=True)
    exec_thread.start()

    robot.exec_and_record_many(traj_gen, N, start_idx=0)

    # Stop executor
    executor.shutdown()
    exec_thread.join(timeout=2)

    rclpy.shutdown()

if __name__ == '__main__':
    main()