import math
from pyquaternion import Quaternion
import numpy as np


"""
# ========================== Control Utility ==========================
"""
def unpack_pose_jacobian_dict(link_names, poses, jacobians):
    poses = np.array(poses)
    jacobians = np.array(jacobians)

    pose_jacobian_dict = {}
    pose_counter = jac_counter = 0
    for link in link_names:
        pose = poses[pose_counter:pose_counter + 3]
        # TODO: remove warning
        jacobian = np.hstack([
            jacobians[jac_counter + 3:jac_counter + 10],
            jacobians[jac_counter + 10:jac_counter + 17],
            jacobians[jac_counter + 17:jac_counter + 24],
        ])
        pose_counter += 3
        jac_counter += 21

        pose_jacobian_dict[link] = [pose, jacobian]
    return pose_jacobian_dict


def wrap_angles(angles):
    # Round the angle, it is still in radiance
    return angles % (2 * np.pi)


def compute_angle_difference(angles1, angles2):
    deltas = np.abs(angles1 - angles2)
    differences = np.minimum(2 * np.pi - deltas, deltas)
    return differences


def check_pose_in_box(pose, safety_box):
    within_box = safety_box.contains(pose)
    return within_box


def convert_quat(q, to="xyzw"):
    """
    Converts quaternion from one convention to another.
    The convention to convert TO is specified as an optional argument.
    If to == 'xyzw', then the input is in 'wxyz' format, and vice-versa.

    Args:
        q: a 4-dim numpy array corresponding to a quaternion
        to: a string, either 'xyzw' or 'wxyz', determining
            which convention to convert to.
    """
    if to == "xyzw":
        return q[[1, 2, 3, 0]]
    if to == "wxyz":
        return q[[3, 0, 1, 2]]
    raise Exception("convert_quat: choose a valid `to` argument (xyzw or wxyz)")


def quat_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.
    >>> q = quat_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> np.allclose(q, [-44, -14, 48, 28])
    True
    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        dtype=np.float32,
    )


def quat_conjugate(quaternion):
    """Return conjugate of quaternion.
    >>> q0 = random_quaternion()
    >>> q1 = quat_conjugate(q0)
    >>> q1[3] == q0[3] and all(q1[:3] == -q0[:3])
    True
    """
    return np.array(
        (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]),
        dtype=np.float32,
    )


def quat_inverse(quaternion):
    """Return inverse of quaternion.
    >>> q0 = random_quaternion()
    >>> q1 = quat_inverse(q0)
    >>> np.allclose(quat_multiply(q0, q1), [0, 0, 0, 1])
    True
    """
    return quat_conjugate(quaternion) / np.dot(quaternion, quaternion)


def euler_to_quat_mul(rotation, quat=None):
    """ Returns a quaternion of a euler rotation """
    q1 = Quaternion(axis=[1, 0, 0], degrees=rotation[0])
    q2 = Quaternion(axis=[0, 1, 0], degrees=rotation[1])
    q3 = Quaternion(axis=[0, 0, 1], degrees=rotation[2])
    q = q3 * q2 * q1
    # q = Quaternion(
    #     convert_quat(mat2quat(euler2mat(np.array(rotation) / 180.0 * np.pi)), to="wxyz")
    # )
    if quat is None:
        final_quat = np.array(list(q), dtype=np.float64)
    else:
        final_quat = np.array(list(Quaternion(quat) * q), dtype=np.float64)
    return final_quat


def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))
    return X, Y, Z