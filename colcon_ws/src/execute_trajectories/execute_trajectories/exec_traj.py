import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from datetime import datetime
import pandas as pd
import numpy as np
import subprocess
import threading
import signal
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

    def __init__(self):
        super().__init__('ur_trajectory_executor')
        # Can be changed to whatever motion controller you want. Just be aware of what action definition the controller uses.
        # passthrough_trajectory_controller uses the same action definition, and might lighten the load on your computer, but does not publish feedback.
        self.trajectory_client = ActionClient(
            self, FollowJointTrajectory, '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        self.trajectory_client.wait_for_server()
        self.lock = threading.Lock()
    
    # Both topics are published at 500 Hz, but rclpy is not necessarily able to keep up with that rate.
    # store the current joint state
    def joint_states_callback(self, msg: JointState):
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        positions_ordered = np.array([msg.position[name_to_index[j]] for j in self.ROBOT_JOINTS])
        
        with self.lock:
            self.curr_joint_states = positions_ordered

    def send_trajectory(self, trajectory: JointTrajectory):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance = Duration(seconds=1).to_msg()
        future = self.trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            print('Trajectory goal rejected')
            return None
        print('Trajectory goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        print('Trajectory execution finished')
        return result_future.result()


    def move_to_point(self, target_point, nsec : float):
        # move the robot to a target point in nsecs

        traj = JointTrajectory()
        traj.joint_names = self.ROBOT_JOINTS
        traj.points = [
            JointTrajectoryPoint(positions = target_point,
                                 velocities = np.zeros((6,), dtype=float),
                    time_from_start = Duration(seconds=nsec).to_msg())
        ]  # allow nsecs to go to home

        return self.send_trajectory(traj)

    
    def exec_and_record(self, traj : JointTrajectory, rosbag_path):
        # read the robot state
        self.curr_joint_states = None

        curr_joint_states_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10
        )

        # Wait until first message arrives
        while self.curr_joint_states is None:
            time.sleep(0.1)

        with self.lock:
            # Unsubscribe
            self.destroy_subscription(curr_joint_states_sub)
            curr_joint_states_sub = None


        # now check whether the robot is close to the start point
        start_point = traj.points[0].positions

        if not np.allclose(self.curr_joint_states, start_point, atol=0.05):
            # robot far away - move to start
            print("Robot far away from start... moving...")
            move_result = self.move_to_point(start_point, 5)  # give 5 seconds to move to start

            if move_result.result.error_code != 0:
                print(f"Failed to move to start: {move_result.result.error_string}")
                return move_result


        # ## if the above step is successful, we can now execute the full trajectory

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
            
            time.sleep(5)  # record before maneuver data

            print("Executing trajectory")
            traj_result = self.send_trajectory(traj)

            print("Recording post-maneuver data")
            time.sleep(3)  # record after maneuver data
        
        finally:
            if rosbag_proc is not None:
                os.killpg(os.getpgid(rosbag_proc.pid), signal.SIGINT)
                rosbag_proc.wait()
            
            print("Done")

        if traj_result.result.error_code != 0:
                print(f"Failed to execute trajectory: {traj_result.result.error_string}")
        
        return traj_result


    @staticmethod
    def get_traj_from_csv_file(file_path):
        times = []
        positions = []
        velocities = []
    
        with open(file_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header

            n_joints = (len(header) - 1) // 2  # first column is time

            for row in reader:
                times.append(float(row[0]))
                pos = [float(x) for x in row[1:1+n_joints]]
                vel = [float(x) for x in row[1+n_joints:1+2*n_joints]]
                positions.append(pos)
                velocities.append(vel)
        
        traj = JointTrajectory()
        traj.joint_names = URTrajectoryExecutor.ROBOT_JOINTS
        traj.points = [
            JointTrajectoryPoint(
                positions=positions[i],
                velocities=velocities[i],
                time_from_start=Duration(seconds=times[i]).to_msg()
            ) for i in range(len(times)) ]
        
        return traj


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_name = f"run_{timestamp}"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    traj_path = os.path.join(script_dir, "traj.csv")
    bag_path = os.path.join(script_dir, "..", "..", "..", "data", "staging", bag_name)
    bag_path = os.path.abspath(bag_path)  # normalize path

    traj = URTrajectoryExecutor.get_traj_from_csv_file(traj_path)


    rclpy.init()
    executor = rclpy.executors.MultiThreadedExecutor()
    robot = URTrajectoryExecutor()
    executor.add_node(robot)

    # Start executor in a separate thread
    exec_thread = threading.Thread(target=executor.spin, daemon=True)
    exec_thread.start()

    robot.exec_and_record(traj, bag_path)

    # Stop executor
    executor.shutdown()
    exec_thread.join(timeout=2)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
