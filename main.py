from deepface import DeepFace
import cv2
import numpy as np
import argparse
from os import listdir
import os.path


ACCEPTED_FORMATS = ['png', 'PNG', 'jpg', 'jpeg', 'JPG', 'JPEG', 'webp']


# Checks if file is legal (supported image type).
def accepted_file(path):
    if not os.path.isfile(path):
        return False
    try:
        suffix = path.split('.')[-1]
    except Exception as e:
        return False
    if suffix in ACCEPTED_FORMATS:
        return True
    else:
        return False


# Finds and returns all face information of faces in an image.
def find_faces(path):
    if not accepted_file(path):
        return None
    return DeepFace.represent(img_path = path)





# Finds target face
def find_target(path):
    faces = find_faces(path)
    if faces is None:
        return None
    largest_size = 0
    for face in faces:
        v = face['embedding']
        fa = face['facial_area']
        if fa['w'] > largest_size:
            size = fa['w']
            best_fit = v

    return np.array(best_fit)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--target', help='Image path containing target face. Largest face in the image gets used.', required=True)
    parser.add_argument('-s','--search', help='Search term to use on Google Images.', required=True)
    parser.add_argument('-n', '--number', help='Number of images to search through.', choices=range(10, 1000), metavar="[10-1000]", default=100)
    args = vars(parser.parse_args())
    print("test")
