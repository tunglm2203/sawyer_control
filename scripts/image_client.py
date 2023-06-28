#!/usr/bin/env python
import rospy

from sawyer_control.srv import image
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
        request = rospy.ServiceProxy(server_name, image, persistent=True)
        response = request()
        return response.image
    except rospy.ServiceException as e:
        print(e)


if __name__ == "__main__":
    for _ in range(1000):
        img = request_image_observation()
        if img is not None:
            img = np.array(img)
            img = img.reshape(480, 480, 3)
            img = img / 255.
            # img = cv2.resize(img, (84, 84))
            # img = img.reshape(84, 84, 3)

            cv2.imshow("CV Image", img)
            cv2.waitKey(5)
