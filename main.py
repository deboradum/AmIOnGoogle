from deepface import DeepFace
import cv2
import numpy as np
import argparse
from os import listdir
import os.path




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--target', help='Image path containing target face. Largest face in the image gets used.', required=True)
    parser.add_argument('-s','--search', help='Search term to use on Google Images.', required=True)
    parser.add_argument('-n', '--number', help='Number of images to search through.', choices=range(10, 1000), metavar="[10-1000]", default=100)
    args = vars(parser.parse_args())
    print("test")
