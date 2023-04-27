from deepface import DeepFace
import cv2
import numpy as np
import argparse
from os import listdir
import os.path
import itertools


ACCEPTED_FORMATS = ['png', 'PNG', 'jpg', 'jpeg', 'JPG', 'JPEG', 'webp']
DEEPFACE_VECTOR_LENGTH = 2622

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


# Creates the matrix of random vectors used for the Locality Sensitive Hashing algorithm.
def create_lsh_matrix(num_vectors):
    m = np.empty((DEEPFACE_VECTOR_LENGTH, num_vectors))
    # Create random vectors and sets these vectors as columns of the matrix.
    for i in range(num_vectors):
        v = np.random.rand(DEEPFACE_VECTOR_LENGTH)
        m[:, i] = v

    return m


# Gets the bucket of faces by multiplying the random vector matrix with the faces matrix.
def get_buckets(faces_matrix):
    # Need to implement check to assert matrix sizes...?

    multiplied = faces_matrix @ lsh_matrix
    # If element is greater than 0, convert to a 1, else to 0.
    r = np.where(multiplied > 0, 1, 0)
    return r


# Converts an array to a string.
# [1, 0, 1, 0, 0] ==> '10100'
def array2string(array):
    s = ''
    for e in array:
        s += str(e)

    return e


if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--target', help='Image path containing target face. Largest face in the image gets used.', required=True)
    parser.add_argument('-q','--query', help='Search term to use on Google Images.', required=True)
    parser.add_argument('-n', '--number', help='Number of images to search through.', choices=range(10, 1000), metavar="[10-1000]", default=100)
    # Need to check what a useful default & max value should be.
    parser.add_argument('-v', '--vector', help='Number of random vectors to use in Locality Sensitive Hashing algorithm.', default=2, choices=range(2, 10), metavar='[2-10]')
    args = vars(parser.parse_args())
    #-------------------------------------------------------------------------#
    global target, search_query, num_images
    target = args['target']
    search_query = args['query']
    num_images = args['number']
    num_rvectors = args['vector']

    # Sets up LSH algorithm variables
    global lsh_matrix
    lsh_matrix = create_lsh_matrix(num_rvectors)
    lsh_bucket_keys = ["".join(seq) for seq in itertools.product("01", repeat=num_rvectors)]
    lsh_buckets = dict()
    for key in lsh_bucket_keys:
        lsh_buckets[key] = np.array([])

    print(lsh_buckets)


    target_face = find_target(target)
    if target_face is None:
        print("No target face found, exiting.")
        exit()
    target_bucket = array2string(get_buckets(target_face))



