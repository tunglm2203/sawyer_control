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

    n_steps = len(data)

    for i in range(n_steps):
        print(f"step: {i}")
        cv2.imshow('image observation', data[i]['observation']['rgb_image'])
        # cv2.waitKey(1)
        key = cv2.waitKey(40) & 0xFF
        if key == ord("q"):
            breakpoint()
        time.sleep(0.4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize demonstrations")
    parser.add_argument("--file", type=str, default="None")
    args = parser.parse_args()
    main(args)