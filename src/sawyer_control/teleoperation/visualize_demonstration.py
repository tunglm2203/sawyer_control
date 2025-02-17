import os
import cv2
import time
import pickle
import argparse

import numpy as np


def main(args):

    with open(args.file, 'rb') as f:
        data = pickle.load(f)

    n_steps = len(data)

    for i in range(n_steps):
        cv2.imshow('image observation', data[i]['observation']['rgb_image'])
        cv2.waitKey(1)
        time.sleep(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize demonstrations")
    parser.add_argument("--file", type=str, default="None")
    args = parser.parse_args()
    main(args)