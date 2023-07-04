#!/usr/bin/env python

from std_msgs.msg import (
    UInt16,
    Header
)
from geometry_msgs.msg import (
    PoseStamped,
    PointStamped,
    Pose,
    Point,
    Quaternion,
)
import rospy

from sensor_msgs.msg import JointState

from intera_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)


"""
Reference:
[1] https://support.rethinkrobotics.com/support/solutions/articles/80000980360
[2] https://github.com/tunglm2203/intera_sdk/blob/b74da757e6b798df12e0d19d1fe05d8c19983f35/intera_interface/src/intera_interface/limb.py#L580
"""

def get_joint_angles(pose, seed_cmd=None, use_advanced_options=False, current=True, tip_name='right_gripper_tip'):
    assert tip_name in ['right_hand', 'right_gripper_tip', 'right_hand_camera']
    limb = "right"
    server_name = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"

    iksvc = rospy.ServiceProxy(server_name, SolvePositionIK)
    ikreq = SolvePositionIKRequest()

    ikreq.pose_stamp.append(pose)
    ikreq.tip_names.append(tip_name)

    if use_advanced_options:
        if current:
            ikreq.seed_mode = ikreq.SEED_CURRENT
        else:
            ikreq.seed_mode = ikreq.SEED_AUTO
        ikreq.seed_angles.append(seed_cmd)

    try:
        rospy.wait_for_service(server_name, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logerr("Service call failed: %s" % (e,))
        return False

    # Check if result valid, and type of seed ultimately used to get solution
    if (resp.result_type[0] > 0):
        limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        return limb_joints
    else:
        rospy.loginfo("INVALID POSE - No Valid Joint Solution Found.")
        raise ValueError


def get_point_stamped(x, y, z):
    point = PointStamped(
        header=Header(stamp=rospy.Time.now(), frame_id='base'),
        point=Point(x=x, y=y, z=z)
    )
    return point


def get_pose_stamped(p_x, p_y, p_z, o_x, o_y, o_z, o_w):
    pose = PoseStamped(
        header=Header(stamp=rospy.Time.now(), frame_id='base'),
        pose=Pose(
            position=Point(x=p_x, y=p_y, z=p_z),
            orientation=Quaternion(x=o_x, y=o_y, z=o_z, w=o_w)
        )
    )
    return pose

def joint_state_from_cmd(cmd):
    js = JointState()
    js.name = cmd.keys()
    js.position = cmd.values()
    return js


def main():
    rospy.init_node("inverse_kinematics_test")
    pose = get_pose_stamped(
        p_x=0.45, p_y=0.16, p_z=0.21,
        o_x=-0.00142460053167, o_y=0.99999999999, o_z=-0.00177030764765, o_w=0.00253311793936
    )
    seed_joint_angles = {
        'right_j6': 1.0, 'right_j5': 1.0, 'right_j4': 0.6,
        'right_j3': 1.5, 'right_j2': -0.7, 'right_j1': -0.76, 'right_j0': 0.2
    }
    target_joint_angles = get_joint_angles(pose, seed_joint_angles, True, False, 'right_gripper_tip')
    print(target_joint_angles)

if __name__ == '__main__':
    main()
