#!/usr/bin/env python3

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import csv
import numpy as np
from datetime import datetime

import rospy
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry

from kf_vio_pnp import VioAugmentedKalmanFilter, KFConfig
from transform import Transform
from pnp.gate_map import GateMap
from pnp.pnp_pose_compose import PnPPoseCompose


class MocapGatekeeper:
    """
    Handles mocap downsampling logic based on the configured mode.
    Modes:
      - continuous:  pass every measurement through
      - downsampled: pass every N-th measurement at target Hz (assumes 100 Hz input)
      - periodic:    alternate available/unavailable windows, downsampled within available windows
    """

    def __init__(self, mode, freq, available_duration, unavailable_duration):
        self.mode = mode
        self.freq = freq
        self.available_duration = available_duration
        self.unavailable_duration = unavailable_duration
        self._count = 0
        self._init_time = None

    def should_skip(self, t):
        if self.mode == 'continuous':
            return False

        elif self.mode == 'downsampled':
            self._count += 1
            if self._count > 10000:
                self._count = 1
            skip_interval = max(1, int(100 * (1 / self.freq)))
            return self._count % skip_interval != 0

        elif self.mode == 'periodic':
            if self._init_time is None:
                self._init_time = t
            elapsed = t - self._init_time
            cycle = self.available_duration + self.unavailable_duration
            position_in_cycle = elapsed % cycle
            if position_in_cycle >= self.available_duration:
                return True
            # Within available window: still apply frequency downsampling
            self._count += 1
            if self._count > 10000:
                self._count = 1
            skip_interval = max(1, int(100 * (1 / self.freq)))
            return self._count % skip_interval != 0

        else:
            rospy.logwarn_once(f"Unknown mocap_mode '{self.mode}', passing all measurements")
            return False


class BiasLogger:
    """Logs VIO bias estimates to a CSV file."""

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._path = os.path.join(log_dir, f'bias_data_{timestamp}.csv')
        self._file = open(self._path, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow(['timestamp', 'bias_x', 'bias_y', 'bias_z'])
        rospy.loginfo(f"Bias logging started: {self._path}")

    def log(self, t, bias):
        self._writer.writerow([t, bias[0], bias[1], bias[2]])
        self._file.flush()

    def close(self):
        try:
            self._file.close()
            rospy.loginfo(f"Bias logging stopped. Data saved to: {self._path}")
        except Exception as e:
            rospy.logerr(f"Failed to close bias log: {e}")


class KFNode:
    """
    ROS1 node that fuses VIO odometry and PnP pose estimates using
    an augmented Kalman filter with online VIO bias estimation.

    Subscriptions:
      /d2vins/odometry        (nav_msgs/Odometry)     – VIO odometry
      /pnp_pose               (geometry_msgs/PoseStamped) – PnP pose
      /mavros/vision_pose/pose (geometry_msgs/PoseStamped) – MoCap as PnP (optional)

    Publication:
      /kf_vio_pnp/odometry    (nav_msgs/Odometry)     – fused estimate (ENU frame)
    """

    def __init__(self):
        rospy.init_node('kf_node')

        # --- Kalman filter config ---
        cfg = KFConfig(
            imu_accel_noise_std=rospy.get_param('~imu_accel_noise_std', 0.2),
            vio_bias_rw_std=rospy.get_param('~vio_bias_rw_std', 0.01),
            vio_pos_std=rospy.get_param('~vio_pos_std', 0.20),
            vio_vel_std=rospy.get_param('~vio_vel_std', 0.30),
            pnp_pos_std=rospy.get_param('~pnp_pos_std', 0.03),
            vio_delay_compensation=rospy.get_param('~vio_delay_compensation', True),
            vio_delay_sec=rospy.get_param('~vio_delay_sec', 0.05),
        )

        # --- Frame transforms ---
        self._transform = Transform(
            vio_yaw_rel_pnp=rospy.get_param('~init_yaw_rad', 0.0),
            vio_translation_rel_pnp=[
                rospy.get_param('~init_pos_x', 0.0),
                rospy.get_param('~init_pos_y', 0.0),
                rospy.get_param('~init_pos_z', 0.0),
            ]
        )
        # --- MoCap gating ---
        self._mocap_gatekeeper = MocapGatekeeper(
            mode=rospy.get_param('~mocap_mode', 'continuous'),
            freq=rospy.get_param('~mocap_freq', 100.0),
            available_duration=rospy.get_param('~mocap_available_duration', 2.0),
            unavailable_duration=rospy.get_param('~mocap_unavailable_duration', 5.0),
        )

        # --- Gate map & PnP pose composer ---
        scene_csv_path = rospy.get_param('~scene_csv_path', 'config/scene.csv')
        self._gate_map = GateMap(Path(scene_csv_path))
        self._pnp_pose_composer = PnPPoseCompose(self._gate_map)

        # --- Internal state ---
        self._kf = VioAugmentedKalmanFilter(cfg)
        self._initialized = False
        self._init_vel = [0.0, 0.0, 0.0]
        self._yaw = 0.0
        self._last_accel_meas = np.zeros(3, dtype=float)
        self._bias_logger = None

        # --- Subscribers ---
        rospy.Subscriber('/d2vins/odometry', Odometry, self._vio_callback, queue_size=10)
        self._gate_pose_sub = rospy.Subscriber('/gate_poses', PoseArray, self._gate_pose_callback, queue_size=10)
        self._mocap_sub     = rospy.Subscriber('/mavros/vision_pose/pose', PoseStamped, self._mocap_callback, queue_size=10)

        # --- Publisher ---
        self._pub_fused = rospy.Publisher('/kf_vio_pnp/odometry', Odometry, queue_size=10)

        rospy.loginfo("KF Node initialized")
        rospy.loginfo(f"Rotation matrix VIO->PnP:\n{self._transform.R_vio_to_pnp}")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_filter(self, t, pos, vel):
        self._kf.init_state(pos=pos, vel=vel, t0=t)
        self._initialized = True
        self._setup_bias_logger(t)
        rospy.loginfo(f"Kalman filter initialised at t={t:.3f}s")

    def _setup_bias_logger(self, t):
        log_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'logs')
        )
        try:
            self._bias_logger = BiasLogger(log_dir)
        except Exception as e:
            rospy.logerr(f"Failed to initialise bias logger: {e}")

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def _vio_callback(self, msg: Odometry):
        t = msg.header.stamp.to_sec()
        pos = [msg.pose.pose.position.x,
               msg.pose.pose.position.y,
               msg.pose.pose.position.z]
        vel = [msg.twist.twist.linear.x,
               msg.twist.twist.linear.y,
               msg.twist.twist.linear.z]
        ori = msg.pose.pose.orientation

        self._yaw = self._transform.quaternion_to_yaw([ori.x, ori.y, ori.z, ori.w])
        pos_tf = self._transform.vio_to_pnp(pos)
        vel_tf = self._transform.vio_to_pnp(vel)
        self._yaw = self._transform.yaw_vio_to_pnp(self._yaw)

        if not self._initialized:
            self._init_filter(t, pos_tf, vel_tf)
            return

        event = {"t": t, "type": "vio", "pos": pos_tf, "vel": vel_tf}
        self._kf.process_event(event, accel_meas=self._last_accel_meas)
        self._publish_fused(t)
        self._log_bias(t)

    def _handle_pnp_measurement(self, t, pos):
        """Shared logic for any PnP-type position measurement."""
        if not self._initialized:
            self._init_filter(t, pos, self._init_vel)
            return
        event = {"t": t, "type": "pnp", "pos": pos}
        self._kf.process_event(event, accel_meas=self._last_accel_meas)
        self._publish_fused(t)
        self._log_bias(t)

    def _gate_pose_callback(self, msg: PoseArray):
        """
        Gate pose array callback.  The array is one-hot: only the non-zero
        element is a valid detection and its index is the gate id.
        """
        t = msg.header.stamp.to_sec()

        for gate_id, pose in enumerate(msg.poses):
            gate_pos_quad = [pose.position.x, pose.position.y, pose.position.z]

            # Skip zero entries (no detection for this gate id)
            if gate_pos_quad[0] == 0.0 and gate_pos_quad[1] == 0.0 and gate_pos_quad[2] == 0.0:
                continue

            gate_pos_np = np.array(gate_pos_quad, dtype=float)
            T_g_to_q = self._pnp_pose_composer.get_T_g_to_q(gate_pos_np)

            rospy.loginfo(f"Gate ID: {gate_id}, Position: {gate_pos_quad}")

            result = self._pnp_pose_composer.comp_quadrotor_pose(gate_id, T_g_to_q)
            quad_pos_world = result.quadrotor_pose_world[:3].tolist()

            self._handle_pnp_measurement(t, quad_pos_world)
            return  # only one valid gate per message

    def _mocap_callback(self, msg: PoseStamped):
        t = msg.header.stamp.to_sec()
        pos = [msg.pose.position.x,
               msg.pose.position.y,
               msg.pose.position.z]
        if self._mocap_gatekeeper.should_skip(t):
            return
        self._handle_pnp_measurement(t, pos)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish_fused(self, t):
        p, v, b, P = self._kf.get_state()
        quat = self._transform.yaw_to_quaternion(self._yaw)

        msg = Odometry()
        msg.header.stamp = rospy.Time.from_sec(t)
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link'

        msg.pose.pose.position.x = float(p[0])
        msg.pose.pose.position.y = float(p[1])
        msg.pose.pose.position.z = float(p[2])

        msg.pose.pose.orientation.x = float(quat[0])
        msg.pose.pose.orientation.y = float(quat[1])
        msg.pose.pose.orientation.z = float(quat[2])
        msg.pose.pose.orientation.w = float(quat[3])

        msg.twist.twist.linear.x = float(v[0])
        msg.twist.twist.linear.y = float(v[1])
        msg.twist.twist.linear.z = float(v[2])

        self._pub_fused.publish(msg)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_bias(self, t):
        if self._bias_logger is None:
            return
        try:
            _, _, b, _ = self._kf.get_state()
            self._bias_logger.log(t, b)
        except Exception as e:
            rospy.logerr(f"Failed to log bias: {e}")

    def stop_bias_logging(self):
        if self._bias_logger is not None:
            self._bias_logger.close()

    # ------------------------------------------------------------------
    # Spin
    # ------------------------------------------------------------------

    def spin(self):
        rospy.spin()


def main():
    node = KFNode()
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.stop_bias_logging()


if __name__ == '__main__':
    main()
