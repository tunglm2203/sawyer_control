import os
import cv2
import time
import pickle
import argparse

import numpy as np


def main(args):

    print(f'Load: {args.file}')
    with open(args.file, 'rb') as f:
        data = pickle.load(f)

    ep_len = len(data)

    for i in range(ep_len - 1):
        print(f"step: {i}")
        img_t = data[i]['observation']['rgb_image']
        img_tp1 = data[i + 1]['observation']['rgb_image']
        img_transition = np.zeros((480, 480 * 2 + 10, 3), dtype=np.uint8)
        img_transition[:, :480, :] = img_t
        img_transition[:, -480:, :] = img_tp1
        cv2.imshow("image transition (t and t+1)", img_transition)
        # cv2.waitKey(1)
        key = cv2.waitKey(40) & 0xFF
        if key == ord("q"):
            breakpoint()
        # if i >= 26:
        #     breakpoint()
        time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize demonstrations")
    parser.add_argument("--file", type=str, default="None")
    args = parser.parse_args()
    main(args)