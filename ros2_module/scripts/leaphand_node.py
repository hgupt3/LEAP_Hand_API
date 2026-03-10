#!/usr/bin/env python3

import logging
import sys
import os
# Add script directory to path so leap_hand_utils is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from leap_hand_utils.dynamixel_client import DynamixelClient
import leap_hand_utils.leap_hand_utils as lhu
from leap_hand.srv import LeapPosition, LeapVelocity, LeapEffort, LeapPosVelEff

_DYN = ParameterDescriptor(dynamic_typing=True)

#LEAP hand conventions:
#180 is flat out home pose for the index, middle, ring, finger MCPs.
#Applying a positive angle closes the joints more and more to curl closed.
#The MCP is centered at 180 and can move positive or negative to that.

#The joint numbering goes from Index (0-3), Middle(4-7), Ring(8-11) to Thumb(12-15) and from MCP Side, MCP Forward, PIP, DIP for each finger.
#For instance, the MCP Side of Index is ID 0, the MCP Forward of Ring is 9, the DIP of Ring is 11

#I recommend you only query when necessary and below 90 samples a second.  Used the combined commands if you can to save time.  Also don't forget about the USB latency settings in the readme.
#The services allow you to always have the latest data when you want it, and not spam the communication lines with unused data.

class LeapNode(Node):
    def __init__(self):
        super().__init__('leaphand_node')
        # Some parameters to control the hand
        self.kP = self.declare_parameter('kP', descriptor=_DYN).get_parameter_value().double_value
        self.kI = self.declare_parameter('kI', descriptor=_DYN).get_parameter_value().double_value
        self.kD = self.declare_parameter('kD', descriptor=_DYN).get_parameter_value().double_value
        self.curr_lim = self.declare_parameter('curr_lim', descriptor=_DYN).get_parameter_value().double_value
        poll_hz = self.declare_parameter('poll_hz', descriptor=_DYN).get_parameter_value().double_value
        self.ema_amount = 0.2
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))

        # Subscribes to a variety of sources that can command the hand
        self.create_subscription(JointState, 'cmd_leap', self._receive_pose, 10)
        self.create_subscription(JointState, 'cmd_allegro', self._receive_allegro, 10)
        self.create_subscription(JointState, 'cmd_ones', self._receive_ones, 10)

        # Publish joint states at a fixed rate (single Dynamixel read, many subscribers)
        self.joint_state_pub = self.create_publisher(JointState, 'leap_joint_states', 10)
        self.joint_names = [f'motor_{i}' for i in range(16)]
        self.create_timer(1.0 / poll_hz, self._publish_joint_states)

        # Creates services that can give information about the hand out
        self.create_service(LeapPosition, 'leap_position', self.pos_srv)
        self.create_service(LeapVelocity, 'leap_velocity', self.vel_srv)
        self.create_service(LeapEffort, 'leap_effort', self.eff_srv)
        self.create_service(LeapPosVelEff, 'leap_pos_vel_eff', self.pos_vel_eff_srv)
        self.create_service(LeapPosVelEff, 'leap_pos_vel', self.pos_vel_srv)
        self.motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        port = self.declare_parameter('port', descriptor=_DYN).get_parameter_value().string_value
        self.get_logger().info(f'Connecting to LEAP Hand on {port}')
        self.dxl_client = DynamixelClient(self.motors, port, 4000000)
        self.dxl_client.connect()

        # Enables position-current control mode and the default parameters.
        # Suppress bus retry noise during init (4Mbps Dynamixel bus is lossy,
        # retries are expected and not actionable).
        dxl_logger = logging.getLogger()
        prev_level = dxl_logger.level
        dxl_logger.setLevel(logging.CRITICAL)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(self.motors, True)
        dxl_logger.setLevel(prev_level)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kP, 84, 2)  # Pgain stiffness
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2)  # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kI, 82, 2)  # Igain
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kD, 80, 2)  # Dgain damping
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2)  # Dgain damping for side to side should be a bit less
        # Max at current (in unit 1ma) so don't overheat and grip too hard
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.curr_lim, 102, 2)
        # Profile Velocity (register 112, 4 bytes) — smooths position moves.
        # Units: 0.229 rev/min per count.  0 = instant (jerky).
        profile_velocity = self.declare_parameter('profile_velocity', descriptor=_DYN).get_parameter_value().integer_value
        if profile_velocity > 0:
            self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * profile_velocity, 112, 4)
            self.get_logger().info(f'Profile Velocity set to {profile_velocity} ({profile_velocity * 0.229:.1f} rev/min, {profile_velocity * 0.02398:.1f} rad/s)')
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
        self.get_logger().info('LEAP Hand connected and ready')

    # Receive LEAP pose and directly control the robot.  Fully open here is 180 and increases in this value closes the hand.
    def _receive_pose(self, msg):
        pose = msg.position
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    #Allegro compatibility, first read the allegro publisher and then convert to leap
    #It adds 180 to the input to make the fully open position at 0 instead of 180.
    def _receive_allegro(self, msg):
        pose = lhu.allegro_to_LEAPhand(msg.position, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # Sim compatibility, first read the sim publisher and then convert to leap
    #Sim compatibility for policies, it assumes the ranges are [-1,1] and then convert to leap hand ranges.
    def _receive_ones(self, msg):
        pose = lhu.sim_ones_to_LEAPhand(np.array(msg.position))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def _publish_joint_states(self):
        """Timer callback: read positions and publish in sim frame (0 = home)."""
        try:
            pos = self.dxl_client.read_pos()
        except Exception:
            return  # transient bus read error, skip this cycle
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = lhu.LEAPhand_to_LEAPsim(pos).tolist()
        self.joint_state_pub.publish(msg)

    # Service that reads and returns the pos of the robot in regular LEAP Embodiment scaling.
    def pos_srv(self, request, response):
        response.position = self.dxl_client.read_pos().tolist()
        return response

    # Service that reads and returns the vel of the robot in LEAP Embodiment
    def vel_srv(self, request, response):
        response.velocity = self.dxl_client.read_vel().tolist()
        return response

    # Service that reads and returns the effort/current of the robot in LEAP Embodiment
    def eff_srv(self, request, response):
        response.effort = self.dxl_client.read_cur().tolist()
        return response
    #Use these combined services to save a lot of latency if you need multiple datapoints
    def pos_vel_srv(self, request, response):
        output = self.dxl_client.read_pos_vel()
        response.position = output[0].tolist()
        response.velocity = output[1].tolist()
        response.effort = np.zeros_like(output[1]).tolist()
        return response
    #Use these combined services to save a lot of latency if you need multiple datapoints
    def pos_vel_eff_srv(self, request, response):
        output = self.dxl_client.read_pos_vel_cur()
        response.position = output[0].tolist()
        response.velocity = output[1].tolist()
        response.effort = output[2].tolist()
        return response

def main(args=None):
    rclpy.init(args=args)
    node = LeapNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        logging.getLogger().setLevel(logging.CRITICAL)
        node.dxl_client.set_torque_enabled(node.motors, False, retries=10)
        node.dxl_client.disconnect()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
