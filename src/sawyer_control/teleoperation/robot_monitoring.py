#!/usr/bin/env python
import rospy

from sawyer_control.srv import type_image
from sawyer_control import PREFIX

import cv2
import numpy as np

"""
This file is used for test image_observation_server, it is not called in launch file.
"""


def request_image_observation():
    server_name = PREFIX + 'image_observation'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_image, persistent=True)
        response = request()
        return response.image
    except rospy.ServiceException as e:
        print(e)


def request_image_overview():
    server_name = PREFIX + 'image_overview'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_image, persistent=True)
        response = request()
        return response.image
    except rospy.ServiceException as e:
        print(e)


if __name__ == "__main__":

    display_height = 480
    display_width = 848

    while True:
        img_observation = request_image_observation()
        img_overview = request_image_overview()

        img_observation = np.array(img_observation)
        img_observation = img_observation.reshape(480, 480, 3)

        background = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        y_offset = (display_height - 480) // 2  # (720 - 480) / 2 = 120
        x_offset = (display_width - 480) // 2  # (1280 - 480) / 2 = 400
        background[y_offset:y_offset + 480, x_offset:x_offset + 480] = img_observation
        image_left = background / 255.


        img_overview = np.array(img_overview)
        img_overview = img_overview.reshape(480, 848, 3)
        img_overview = cv2.resize(img_overview, (display_width, display_height))
        image_right = img_overview / 255.

        img = np.hstack([image_left, image_right])

        cv2.imshow("Robot Monitoring (LEFT: Image Observation, RIGHT: Image Overview)", img)
        key = cv2.waitKey(40) & 0xFF
        if key == ord("q"):
            break

