# AmIOnGoogle
A tool that uses DeepFace &amp; Random Projection Hasing that checks if your face is on Google Images

To gain more experience with AI & my Jetson Nano, I created this tool. The tool lets you select an image and a search term. The biggest face from the input image is analyzed and vectorized by DeepFace AI. Next, it searches Google Images and uses DeepFace to detect faces in the images. A locality sensitive hashing algorithm is then applied on these face-feature vectors to bucketize the faces. Then, the faces in the same bucket as the input image are further compared using cosine similarity to find the same person in the Google Images.

# Locality Sensitive Hashing
Locality sensitive hashing is a hashing algorithm that maps similar vectors to the same bucket. There are many different algorithms for this, but the algorithm I used is *Random Projection Hashing*. Here, the dot product of a to be mapped d-dimensional vector is taken with k random vectors. If the dot product of a random vector and the input vector is positive, it is mapped to a 1. If it is zero or negative, it is mapped to a 0. The k results are then appended and the bitstring of length k is the bucket this input vector belongs to. This method ensures that similar vectors end up in the same bucket, which means only one (or a few) buckets have to be analyzed further. You can read more about this algorithm [here](http://benwhitmore.altervista.org/simhash-and-solving-the-hamming-distance-problem-explained/).

# Demo
WIP
