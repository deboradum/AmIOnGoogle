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
from serpapi import GoogleSearch
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get('APIKEY')

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

    try:
        return DeepFace.represent(img_path = path)
    except Exception as e:
        return None


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
    multiplied = faces_matrix.T @ lsh_matrix
    # If element is greater than 0, convert to a 1, else to 0.
    r = np.where(multiplied > 0, 1, 0)
    return r


# Converts an array to a string, for example:
# [1, 0, 1, 0, 0] ==> '10100'
def array2string(array):
    s = ''
    for e in array:
        s += str(e)

    return s


# Converts an array of face dicts ({embedding: [], facial_area: []}[]) to a matrix of faces.
def faces2matrix(faces):
    num_faces = len(faces)
    m = np.empty((DEEPFACE_VECTOR_LENGTH, num_faces))
    for i, face in enumerate(faces):
        v = np.array(face['embedding'])
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


# Gets image urls using SerpAPI. TODO
def get_image_urls(query, num_results, page=1):
    links = []
    # Creates & sends the query.
    params = {
        "engine": "google_images",
        "q": query,
        "api_key": api_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    im_results = results['images_results']
    # Gets image links.
    for i, im_json in enumerate(im_results):
        if i > num_results:
            break
        link = im_json['original']
        links.append(link)
    # If more than 100 results should be received, get more from the next page.
    if num_results > 100:
        more_links = get_image_urls(query, num_results-100, page+1)
        links += more_links
    
    return links

# Checks if a face is similar (enough) to the main face.
# Accepted modes: cosine or euclidean.
def is_similar(target_v, face_v, threshold):
    cs = np.dot(target_v, face_v) / (np.linalg.norm(target_v) * np.linalg.norm(face_v))
    if cs >= threshold:
        return True
    else:
        return False


def num_checker(num):
    n = int(num)

    if n >= 10 and n < 1000:
        return n
    
    raise argparse.ArgumentTypeError('Invalid choice, choose between 10 and 1000.')

if  __name__ == '__main__':
    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--target', help='Image path containing target face. Largest face in the image gets used.', required=True)
    parser.add_argument('-q','--query', help='Search term to use on Google Images.', required=True)
    parser.add_argument('-n', '--number', help='Number of images to search through.', type=num_checker, default=100)
    # Need to check what a useful default & max value should be.
    parser.add_argument('-v', '--vector', help='Number of random vectors to use in Locality Sensitive Hashing algorithm.', default=1, choices=range(1, 10), metavar='[1-10]')
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

    print("Finding target face")

    # Sets target face variables.
    target_face = find_target(target)
    if target_face is None:
        print("No target face found, exiting.")
        exit()
    target_bucket = array2string(get_buckets(target_face))

    print('Found target face')

    # Goes over all images, finds faces, bucketizes them and inserts them into the lsh dictionary.
    img_urls = get_image_urls(search_query, num_images)
    for url in img_urls:
        try:
            get_image(url)
        except Exception as e:
            delete_image()
            continue
        print("Checking image...")
        faces = find_faces('image.jpg')
        if faces is None:
            print('No faces found')
            continue
        print('Number of faces found in image:', len(faces))
        faces_m = faces2matrix(faces)
        bucket_m = get_buckets(faces_m)
        for i, b in enumerate(bucket_m):
            bucket = array2string(b)
            # Skips vectors that are not in the same bucket as the target face.
            # This saves space, but is not neccesary.
            if bucket != target_bucket:
                continue
            lsh_buckets[bucket].append((faces[i]['embedding'], url))
            print("Found similar face")
        delete_image()
       
    potential_faces = lsh_buckets[target_bucket]
    print(f'bucketizd all faces, {len(potential_faces)} in the same bucket as target.')
    f = open('faces.txt', 'w+')
    fw = open('unmatched.txt', 'w+')
    num_same = 0
    for face_v, url in potential_faces:
        if is_similar(target_face, face_v, 0.65):
            f.write(f'{url}\n')
            num_same += 1
        else:
            fw.write(f'{url}\n')
    print(f"Checked all faces. {num_same} identical found.")
    f.close()
    fw.close()




