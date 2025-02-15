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


if __name__ == "__main__":
    for _ in range(10000):
        img = request_image_observation()
        if img is not None:
            img = np.array(img)
            # import pdb; pdb.set_trace()
            # img = img.reshape(240, 424, 3)
            #img = img.reshape(480, 640, 3)
            #img = img.reshape(480, 480, 3)
            #img = img.reshape(400, 400, 3)
            # img = img.reshape(300, 300, 3)
            img = img.reshape(480, 480, 3)
            #img = img.reshape(720, 1280, 3)
            #cv2.imwrite("/home/tung/ros_ws/offline_dataset/test0126.jpg",img)
            img = img / 255.
            #img = cv2.resize(img, (84, 84))
            # img = img.reshape(84, 84, 3)
      
            cv2.imshow("CV Image", img)
            cv2.waitKey(1)
            
