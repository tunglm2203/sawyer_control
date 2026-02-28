#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
import rospy
import tf
import tf.transformations as tf_trans
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from dt_apriltags import Detector

class StationaryWorldNode:
    def __init__(self):
        rospy.init_node("realsense_stationary_world")
        self.bridge = CvBridge()
        self.tf_broadcaster = tf.TransformBroadcaster()

        # --- Configuration ---
        self.ref_tag_id = 8  
        self.tag_configs = {
            1: {"size": 0.03,  "name": "upper_drawer", "color": (1, 0, 0)},
            2: {"size": 0.05,  "name": "gripper",      "color": (0, 1, 0)},
            3: {"size": 0.04,  "name": "plate",        "color": (0, 0, 1)},
            4: {"size": 0.015, "name": "banana",       "color": (1, 1, 0)},
            5: {"size": 0.03,  "name": "arm",          "color": (1, 0, 1)},
            6: {"size": 0.02,  "name": "red_box",      "color": (1, 0.2, 0)},
            7: {"size": 0.02,  "name": "green_box",    "color": (0.2, 1, 0)},
            8: {"size": 0.09,  "name": "table",        "color": (0.5, 0.5, 0.5)}
        }

        self.is_calibrated = False
        self.t_world_to_cam = None 

        # --- Publishers ---
        # 1. Vision Publishers (For Dataset)
        self.color_pub = rospy.Publisher("/camera/color/image_raw", Image, queue_size=1)
        self.depth_pub = rospy.Publisher("/camera/depth/image_rect_raw", Image, queue_size=1)
        self.color_info_pub = rospy.Publisher("/camera/color/camera_info", CameraInfo, queue_size=1)
        
        # 2. Annotation & Marker Publishers
        self.image_pub = rospy.Publisher("/apriltag/image_annotated", Image, queue_size=1)
        self.marker_pub = rospy.Publisher("/apriltag/3d_pose_markers", MarkerArray, queue_size=10)
        self.pose_stamped_pub = rospy.Publisher("/apriltag/3d_pose", PoseStamped, queue_size=10)

        # --- Detector & Camera Setup ---
        self.at_detector = Detector(families='tag36h11', nthreads=4, quad_decimate=1.0, refine_edges=1)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Standard Resolution for high-quality VLA data
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color) # Align depth to color
        
        # Get static intrinsics for CameraInfo
        self.color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = self.color_stream.get_intrinsics()
        self.camera_params = [self.intrinsics.fx, self.intrinsics.fy, self.intrinsics.ppx, self.intrinsics.ppy]

    def get_camera_info(self, stamp):
        """Generates standard ROS CameraInfo message."""
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = "camera_color_optical_frame"
        msg.width, msg.height = self.intrinsics.width, self.intrinsics.height
        msg.distortion_model = "plumb_bob"
        msg.D = list(self.intrinsics.coeffs)
        msg.K = [self.intrinsics.fx, 0, self.intrinsics.ppx, 0, self.intrinsics.fy, self.intrinsics.ppy, 0, 0, 1]
        msg.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        msg.P = [self.intrinsics.fx, 0, self.intrinsics.ppx, 0, 0, self.intrinsics.fy, self.intrinsics.ppy, 0, 0, 0, 1, 0]
        return msg

    def make_4x4_matrix(self, tvec, Rmat):
        T = np.eye(4)
        T[:3, :3] = Rmat
        T[:3, 3] = tvec.reshape(3)
        return T

    def create_marker_pair(self, tag_id, name, pos, color):
        sphere = Marker()
        sphere.header.frame_id = "world"
        sphere.header.stamp = rospy.Time.now()
        sphere.ns, sphere.id, sphere.type = "shapes", tag_id, Marker.SPHERE
        sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = pos
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.02
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = color[0], color[1], color[2], 1.0
        sphere.lifetime = rospy.Duration(0.2)

        label = Marker()
        label.header = sphere.header
        label.ns, label.id, label.type = "labels", tag_id + 100, Marker.TEXT_VIEW_FACING
        label.text = name
        label.pose.position.x, label.pose.position.y, label.pose.position.z = pos[0], pos[1], pos[2] + 0.03
        label.scale.z = 0.02
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.lifetime = rospy.Duration(0.2)
        return sphere, label

    def run(self):
        rospy.loginfo("Stationary World Node Started with Full Vision Streaming.")
        try:
            while not rospy.is_shutdown():
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame: continue

                stamp = rospy.Time.now()
                color_img = np.asanyarray(color_frame.get_data())
                depth_img = np.asanyarray(depth_frame.get_data())
                gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

                # --- Publish Raw RealSense Data ---
                self.color_pub.publish(self.bridge.cv2_to_imgmsg(color_img, "bgr8"))
                self.depth_pub.publish(self.bridge.cv2_to_imgmsg(depth_img, "16UC1"))
                self.color_info_pub.publish(self.get_camera_info(stamp))

                # AprilTag Detection
                tags = self.at_detector.detect(gray, estimate_tag_pose=True, 
                                             camera_params=self.camera_params, tag_size=0.05)
                marker_array = MarkerArray()

                if not self.is_calibrated:
                    for tag in tags:
                        if tag.tag_id == self.ref_tag_id:
                            actual_t = tag.pose_t * (0.09 / 0.05)
                            self.t_world_to_cam = np.linalg.inv(self.make_4x4_matrix(actual_t, tag.pose_R))
                            self.is_calibrated = True
                    cv2.putText(color_img, "WAITING FOR TAG 8...", (40, 40), 1, 2, (0, 0, 255), 2)
                    rospy.loginfo("WAITING FOR TAG 8...")
                else:
                    for tag in tags:
                        if tag.tag_id not in self.tag_configs: continue
                        cfg = self.tag_configs[tag.tag_id]
                        t_cam_to_tag = self.make_4x4_matrix(tag.pose_t * (cfg["size"] / 0.05), tag.pose_R)
                        t_world_to_tag = np.dot(self.t_world_to_cam, t_cam_to_tag)

                        pos = t_world_to_tag[:3, 3]
                        quat = tf_trans.quaternion_from_matrix(t_world_to_tag)
                        
                        self.tf_broadcaster.sendTransform(pos, quat, stamp, cfg["name"], "world")
                        s, l = self.create_marker_pair(tag.tag_id, cfg["name"], pos, cfg["color"])
                        marker_array.markers.extend([s, l])

                        ps = PoseStamped()
                        ps.header.stamp, ps.header.frame_id = stamp, cfg["name"]
                        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = pos
                        ps.pose.orientation.w = 1.0
                        self.pose_stamped_pub.publish(ps)
                        
                        self.visualize(color_img, tag, cfg["name"])
                    self.marker_pub.publish(marker_array)

                self.image_pub.publish(self.bridge.cv2_to_imgmsg(color_img, "bgr8"))
        finally:
            self.pipeline.stop()

    def visualize(self, img, tag, name):
        pts = tag.corners.astype(int)
        for i in range(4): cv2.line(img, tuple(pts[i]), tuple(pts[(i+1)%4]), (0, 255, 0), 2)
        cv2.putText(img, name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

if __name__ == "__main__":
    StationaryWorldNode().run()