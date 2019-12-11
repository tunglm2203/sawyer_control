#!/usr/bin/env python
import rospy
from sawyer_control.srv import image
import numpy as np


def request_observation():
    rospy.init_node('ba')
    rospy.wait_for_service('images')
    try:
        get_image = rospy.ServiceProxy('images', image, persistent=True)
        obs = get_image()

        return (
            obs.image
        )
    except rospy.ServiceException as e:
        print(e)


if __name__ == "__main__":
    import cv2
    for _ in range(10000):
        img = request_observation()
        img = np.array(img)
        img = img.reshape(480, 480, 3)
        img = img / 255.
        img = cv2.resize(img, (84, 84))
        img = img.reshape(84, 84, 3)

        cv2.imshow("CV Image", img)
        cv2.waitKey(5)
