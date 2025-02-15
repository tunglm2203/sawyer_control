#!/usr/bin/env python
import rospy

from sawyer_control.srv import type_image, type_imageResponse
from sawyer_control import PREFIX

from sensor_msgs.msg import Image as Image_msg
import cv2
from cv_bridge import CvBridge, CvBridgeError
import copy
import thread
import numpy as np


class Latest_observation(object):
    def __init__(self):
        # color image:
        self.img_cv2 = None
        self.img_cropped = None
        self.img_msg = None

        # depth image:
        self.d_img_raw_npy = None  # 16 bit raw data
        self.d_img_cropped_npy = None
        self.d_img_cropped_8bit = None
        self.d_img_msg = None


class RealSenseRecorder(object):
    def __init__(self):
        rospy.Subscriber("/camera_overview/color/image_raw", Image_msg, self.store_latest_image)

        self.ltob = Latest_observation()
        # self.ltob_aux1 = Latest_observation()

        self.bridge = CvBridge()

        def spin_thread():
            rospy.spin()

        thread.start_new(spin_thread, ())

    def store_latest_image(self, data):
        self.ltob.img_msg = data
        # Supported resolution:
        # WxHxC = (1280, 720, 3)
        # WxHxC = (640, 480, 3)
        # WxHxC = (424, 240, 3)
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        # self.ltob.img_cv2 = cv_image  # (84, 84)
        self.ltob.img_cv2 = self.crop_highres(cv_image)  # (84, 84)

    def crop_highres(self, cv_image):
        # startcol = 80
        # startrow = 0
        # endcol = startcol + 480
        # endrow = startrow + 480
        # cv_image = copy.deepcopy(cv_image[startrow:endrow, startcol:endcol])
        # cv_image = cv_image[startrow:endrow, startcol:endcol]
        # cv_image = cv2.resize(cv_image, (250, 250))
        return cv_image


def handle_get_image_observation(request):
    img = cam.ltob.img_cv2
    img = np.array(img)
    image = img.flatten().tolist()
    return type_imageResponse(image)


def image_observation_server():
    node_name = PREFIX + 'image_overview_server'
    server_name = PREFIX + 'image_overview'
    rospy.init_node(node_name)

    global cam
    # You should choose the corresponding camera
    # cam = KinectRecorder()
    cam = RealSenseRecorder()
    # cam = LogitechRecorder()
    # cam = KinectSimRecorder()

    server = rospy.Service(server_name, type_image, handle_get_image_observation)
    rospy.spin()


if __name__ == "__main__":
    image_observation_server()
