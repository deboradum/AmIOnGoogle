# AmIOnGoogle
A tool that uses DeepFace &amp; Random Projection Hashing that checks if your face is on Google Images

To gain more experience with AI & my Jetson Nano, I created this tool. The tool lets you select an image and a search term. The biggest face from the input image is analyzed and vectorized by DeepFace AI. Next, it searches Google Images and uses DeepFace to detect faces in the images. A locality sensitive hashing algorithm is then applied on these face-feature vectors to bucketize the faces. Then, the faces in the same bucket as the input image are further compared using cosine similarity to find the same person in the Google Images.

[SerpAPI](https://serpapi.com) is used to obtain the images. A free account can be made and used for 100 queries. 100 images can be searched for per query. For more queries, a paid subscription is possible.

# Locality Sensitive Hashing
Locality sensitive hashing is a hashing algorithm that maps similar vectors to the same bucket. There are many different algorithms for this, but the algorithm I used is *Random Projection Hashing*. Here, the dot product of a to be mapped d-dimensional vector is taken with k random vectors. If the dot product of a random vector and the input vector is positive, it is mapped to a 1. If it is zero or negative, it is mapped to a 0. The k results are then appended and the bitstring of length k is the bucket this input vector belongs to. This method ensures that similar vectors end up in the same bucket, which means only one (or a few) buckets have to be analyzed further. You can read more about this algorithm [here](http://benwhitmore.altervista.org/simhash-and-solving-the-hamming-distance-problem-explained/).

# Usage
The tool can be used as follows:
```
python3 main.py -t '<target_img_path>' -q '<search_query>' [-n <number_of_results>] [-v <number_of_vectors>]
```
With parameters:
```
-t TARGET, --target TARGET    Image path containing target face. Largest face in the image gets used.
-q QUERY, --query QUERY       Search term to use on Google Images.
-n NUMBER, --number NUMBER    Number of images to search through.
-v, --vector [1-10]    Number of random vectors to use in Locality Sensitive Hashing algorithm.
```

The default number of random vectors to use is one. The reason for this, is because any more will lead to more false negatives, thus decreasing the accuracy of the program. More random vecctors is of course possible and will lead to less false positives, which means a faster result. The quality of the result will be worse, though.


# Demo
https://user-images.githubusercontent.com/88938032/235716391-02d238b7-1b0e-4f90-8df6-34c5064e01d2.mov
