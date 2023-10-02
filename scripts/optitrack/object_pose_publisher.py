#! /usr/bin/env python

import rospy
import tf
import numpy as np
from std_msgs.msg import Header
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion
)


N_CORNERS = 4

def object_pose_publisher():
    rospy.init_node('objects_pose_publisher')

    global table_center_pose_pub
    global table_corner_pose_pubs

    # The commented out bellow for creating publisher for 8 edges
    # publishers = [rospy.Publisher(f'corner_point_{i}', PointStamped, queue_size=10) for i in range(8)]
    table_center_pose_pub = rospy.Publisher('/mocap_node/table_top/center_pose', PoseStamped, queue_size=10)
    table_corner_pose_pubs = [rospy.Publisher('/mocap_node/table_top/corner_pose_{}'.format(i), PoseStamped, queue_size=10) for i in range(N_CORNERS)]


def send_object_pose(pose_msg):
    """
    Notation: the structure of table is shown as figure below:
        c0, c1, c2, c3: corner 0, corner 1, corner 2, corner 3
        (c): center of tabletop

    Diagram for tabletop:    
        c0___________________________________________________c1
        |                                                    |\ dz
        |                                                    | \ 
        |dx                     (c)                          |  |
        |                                                    |  |
        |                                                    |  |
        c2_______________________dy__________________________c3 |
        \                                                     \ |
         \_____________________________________________________\|
    """

    center_position = np.array([
        pose_msg.pose.position.x,
        pose_msg.pose.position.y,
        pose_msg.pose.position.z
    ])
    center_orientation = np.array([
        pose_msg.pose.orientation.x,
        pose_msg.pose.orientation.y,
        pose_msg.pose.orientation.z,
        pose_msg.pose.orientation.w
    ])

    # The physical parameter of table (meter)
    dx, dy, dz = 0.257, 0.65, 0.037

    # Rotation matrix calculation
    rotation_matrix = tf.transformations.quaternion_matrix(center_orientation)[:3, :3]

    # Edge coordinate calculation
    half_dims = [dx / 2, dy / 2, dz / 2]
    offsets = np.array([
        [-1, -1, 0],    # c0
        [-1, 1, 0],     # c1
        [1, -1, 0],     # c2
        [1, 1, 0],      # c3
    ])
    '''
    offsets = np.array([
        [-1, -1, -1], 
        [1, -1, -1],
        [-1, 1, -1],
        [1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1]
    ])
    '''

    #corners = [center_position + np.dot(rotation_matrix, np.multiply(half_dims, offset)) for offset in offsets]
    corner_positions = [center_position + np.multiply(half_dims, offset) for offset in offsets]

    # Publish corner's pose
    for i, corner_position in enumerate(corner_positions):
        corner_pose_msg = PoseStamped(
            header=pose_msg.header,
            pose=Pose(
                position=Point(
                    x=corner_position[0],
                    y=corner_position[1],
                    z=corner_position[2]
                ),
                orientation=Quaternion(
                    x=center_orientation[0],
                    y=center_orientation[1],
                    z=center_orientation[2],
                    w=center_orientation[3],
                ),
            ),
        )
        table_corner_pose_pubs[i].publish(corner_pose_msg)

    # Publish center's pose: This is actually same with original table top from mocap_optitrack
    center_pose_msg = PoseStamped(
            header=pose_msg.header,
            pose=Pose(
                position=Point(
                    x=center_position[0],
                    y=center_position[1],
                    z=center_position[2]
                ),
                orientation=Quaternion(
                    x=center_orientation[0],
                    y=center_orientation[1],
                    z=center_orientation[2],
                    w=center_orientation[3],
                ),
            ),
        )
    table_center_pose_pub.publish(center_pose_msg)


if __name__ == '__main__':
    try:
        object_pose_publisher()     # Constructing Publisher
        rospy.Subscriber('/mocap_node/table_top/pose', PoseStamped, send_object_pose)  # Constructing Subscriber
        rospy.spin()
    except rospy.ROSInterruptException as e:
        print(e)