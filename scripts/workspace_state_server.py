#!/usr/bin/env python
import rospy
import intera_interface

import copy
import time
import numpy as np

from sawyer_control.srv import (
    type_object_pose, type_object_poseResponse,
)
from sawyer_control import PREFIX

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion
)


class Furniture_table(object):
    def __init__(self):
        rospy.Subscriber("/mocap_node/table_top/corner_pose_0", PoseStamped, self.handle_table_corner0_pose)
        rospy.Subscriber("/mocap_node/table_top/corner_pose_1", PoseStamped, self.handle_table_corner1_pose)
        rospy.Subscriber("/mocap_node/table_top/corner_pose_2", PoseStamped, self.handle_table_corner2_pose)
        rospy.Subscriber("/mocap_node/table_top/corner_pose_3", PoseStamped, self.handle_table_corner3_pose)

        rospy.Subscriber("/mocap_node/leg_black/pose", PoseStamped, self.handle_leg_black_pose)
        rospy.Subscriber("/mocap_node/leg_blue/pose", PoseStamped, self.handle_leg_blue_pose)
        rospy.Subscriber("/mocap_node/leg_white/pose", PoseStamped, self.handle_leg_white_pose)
        rospy.Subscriber("/mocap_node/leg_pink/pose", PoseStamped, self.handle_leg_pink_pose)

        self.table_corner0_pose = None
        self.table_corner1_pose = None
        self.table_corner2_pose = None
        self.table_corner3_pose = None

        self.leg_black_pose = None
        self.leg_blue_pose = None
        self.leg_white_pose = None
        self.leg_blue_pose = None

    def handle_table_corner0_pose(self, data):
        self.table_corner0_pose = np.array([
            data.pose.position.x, data.pose.position.y, data.pose.position.z,
            data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w
        ], dtype=np.float64)

    def handle_table_corner1_pose(self, data):
        self.table_corner1_pose = np.array([
            data.pose.position.x, data.pose.position.y, data.pose.position.z,
            data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w
        ], dtype=np.float64)

    def handle_table_corner2_pose(self, data):
        self.table_corner2_pose = np.array([
            data.pose.position.x, data.pose.position.y, data.pose.position.z,
            data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w
        ], dtype=np.float64)

    def handle_table_corner3_pose(self, data):
        self.table_corner3_pose = np.array([
            data.pose.position.x, data.pose.position.y, data.pose.position.z,
            data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w
        ], dtype=np.float64)

    def handle_leg_black_pose(self, data):
        self.leg_black_pose = np.array([
            data.pose.position.x, data.pose.position.y, data.pose.position.z,
            data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w
        ], dtype=np.float64)

    def handle_leg_blue_pose(self, data):
        self.leg_blue_pose = np.array([
            data.pose.position.x, data.pose.position.y, data.pose.position.z,
            data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w
        ], dtype=np.float64)

    def handle_leg_white_pose(self, data):
        self.leg_white_pose = np.array([
            data.pose.position.x, data.pose.position.y, data.pose.position.z,
            data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w
        ], dtype=np.float64)

    def handle_leg_pink_pose(self, data):
        self.leg_pink_pose = np.array([
            data.pose.position.x, data.pose.position.y, data.pose.position.z,
            data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w
        ], dtype=np.float64)


# Note: do not remove input arg "request", it is required for ROS
def handle_get_observation(request):
    table_corner0 = copy.copy(furniture_table.table_corner0_pose)
    table_corner1 = copy.copy(furniture_table.table_corner1_pose)
    table_corner2 = copy.copy(furniture_table.table_corner2_pose)
    table_corner3 = copy.copy(furniture_table.table_corner3_pose)

    leg_black = copy.copy(furniture_table.leg_black_pose)
    leg_blue = copy.copy(furniture_table.leg_blue_pose)
    leg_white = copy.copy(furniture_table.leg_white_pose)
    leg_pink = copy.copy(furniture_table.leg_pink_pose)

    return type_object_poseResponse(table_corner0, table_corner1, table_corner2, table_corner3,
                                    leg_black, leg_blue, leg_white, leg_pink)

def workspace_observation_server():
    node_name = PREFIX + 'workspace_observation_server'
    object_state_server_name = PREFIX + 'object_state_server'
    rospy.init_node(node_name)

    global furniture_table
    furniture_table = Furniture_table()

    sv1 = rospy.Service(object_state_server_name, type_object_pose, handle_get_observation)

    rospy.spin()


if __name__ == "__main__":
    workspace_observation_server()
