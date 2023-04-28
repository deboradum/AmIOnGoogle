from deepface import DeepFace
import cv2
import numpy as np
import argparse
from os import listdir
import os.path
import itertools
from PIL import Image
import requests
from io import BytesIO


ACCEPTED_FORMATS = ['png', 'PNG', 'jpg', 'jpeg', 'JPG', 'JPEG', 'webp']
DEEPFACE_VECTOR_LENGTH = 2622


TEST_URLS = ['https://www.hollywoodreporter.com/wp-content/uploads/2022/10/GettyImages-1398676954-H-2022.jpg?w=1296', 
             'https://parade.com/.image/t_share/MTk3NDgxMjQzNjIzODI2NDM4/kanye-west-wife-kanye-west-bianca-censori.jpg',
             'https://media.gq-magazine.co.uk/photos/5d13a7d52881ccbf300a9586/16:9/pass/kanye-west-03-gq-7feb19_getty_b.jpg',
             'https://images0.persgroep.net/rcs/01FOn6gfdTOChv-rnFORx_DpEJM/diocontent/160165041/_fitwidth/763?appId=93a17a8fd81db0de025c8abd1cca1279&quality=0.8',
             'https://upload.wikimedia.org/wikipedia/commons/b/bb/Kanye_West_at_the_Met_Gala_in_2019_2.png',
             'https://ca-times.brightspotcdn.com/dims4/default/6f80eaa/2147483647/strip/false/crop/2514x1556+0+0/resize/1486x920!/quality/80/?url=https%3A%2F%2Fcalifornia-times-brightspot.s3.amazonaws.com%2F64%2F2d%2Fb889abb448c695d9afcd1afc378e%2Fkim-kardashian-kanye-west-divorce-71287.jpg',
             'https://assets.euromoneydigital.com/dims4/default/53d1382/2147483647/strip/true/crop/620x380+0+0/resize/840x515!/quality/90/?url=http%3A%2F%2Feuromoney-brightspot.s3.amazonaws.com%2F06%2F30%2Faf0e9e02463a8e05ab6ece2153e1%2Fkanye-west-adidas-logo-2023.png',
             'https://wwd.com/wp-content/uploads/2019/11/kanye-west-at-fast-company-panel-nyc.jpg?w=1000&h=563&crop=1',
             'https://images.theconversation.com/files/110255/original/image-20160204-5857-3nvdk7.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=1200&h=1200.0&fit=crop',
             'https://failurebeforesuccess.com/wp-content/uploads/2022/07/1099403-scaled.jpg']


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


# Converts an array to a string, for example:
# [1, 0, 1, 0, 0] ==> '10100'
def array2string(array):
    s = ''
    for e in array:
        s += str(e)

    return e


# Converts an array of face dicts ({embedding: [], facial_area: []}[]) to a matrix of faces.
def faces2matrix(faces):
    num_faces = len(faces)
    m = np.empty((DEEPFACE_VECTOR_LENGTH, num_faces))
    for i, face in enumerate(faces):
        v = np.array(v['embedding'])
        m[:, i] = v

    return m


# Gets and saves image from url to hard drive. Returns 1 on failure, 0 otherwise.
def get_image(url):
    r = requests.get(url)
    if r.status_code != 200:
        print("Could not get image:", url)
        return 1
    im = Image.open(BytesIO(r.content))

    # Need to find proper suffix for each file
    im.save('image.jpg')

    return 0


# Deletes image from drive.
def delete_image():
    if os.path.isfile('image.jpg'):
        os.remove('image.jpg')


# Checks if a face is similar (enough) to the main face.
# Accepted modes: cosine or euclidean.
def is_similar(check_face, threshold):
    cs = np.dot(self.targetFaceV, check_face) / (np.linalg.norm(self.targetFaceV) * np.linalg.norm(check_face))
    if cs >= threshold:
        return True
    else:
        return False


if  __name__ == '__main__':
    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--target', help='Image path containing target face. Largest face in the image gets used.', required=True)
    parser.add_argument('-q','--query', help='Search term to use on Google Images.', required=True)
    parser.add_argument('-n', '--number', help='Number of images to search through.', choices=range(10, 1000), metavar="[10-1000]", default=100)
    # Need to check what a useful default & max value should be.
    parser.add_argument('-v', '--vector', help='Number of random vectors to use in Locality Sensitive Hashing algorithm.', default=2, choices=range(2, 10), metavar='[2-10]')
    args = vars(parser.parse_args())

    # Sets variables based on arguments.
    global target, search_query, num_images
    target = args['target']
    search_query = args['query']
    num_images = args['number']
    num_rvectors = args['vector']

    #-------------------------------------------------------------------------#

    # Sets up LSH algorithm variables.
    global lsh_matrix
    lsh_matrix = create_lsh_matrix(num_rvectors)
    lsh_bucket_keys = ["".join(seq) for seq in itertools.product("01", repeat=num_rvectors)]
    lsh_buckets = {key: [] for key in lsh_bucket_keys}

    # Sets target face variables.
    target_face = find_target(target)
    if target_face is None:
        print("No target face found, exiting.")
        exit()
    target_bucket = array2string(get_buckets(target_face))

    # Goes over all images, finds faces, bucketizes them and inserts them into the lsh dictionary.
    img_urls = get_image_urls(search_query, num_images)
    for url in img_urls:
        get_image(TEST_URLS[0])
        faces = find_faces('image.jpg')
        faces_m = faces2matrix(faces)
        bucket_m = get_buckets(faces_m)
        for i, b in enumerate(bucket_m):
            bucket = array2string(b)
            # Skips vectors that are not in the same bucket as the target face.
            # This saves space, but is not neccesary.
            if bucket != target_bucket:
                continue
            lsh_buckets[bucket].append((faces_m[i], url))
       

    potential_faces = lsh_buckets[target_face]
    for face_v, url in potential_faces:
        if is_similar(face_v, 0.65)  # TODO: variable threshold
            print('Found similar face!', url)
    






