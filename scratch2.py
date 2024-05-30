import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
from bisect import bisect_left
import torch.nn.functional as F
import torch

BETA = 0.00002
DELTA = 5  # resolution constant for counting two colors as the same
NU = 8  # when querying, fraction of pattern vector that is not hidden
ZETA = 10  # Number of randomly generated vectors per n
ETA = 1 # number of vectors per cluster

# returns a float between 0 and 1 denoting the closeness of two vectors
def normalize_dot(x, y):
    res = np.dot(x, y)
    mag_x = np.sqrt(np.dot(x, x))
    mag_y = np.sqrt(np.dot(y, y))
    res = res / (mag_x * mag_y)
    return res


# Ramsauer's update rule
def update_rule(z, patterns, BETA):
    return patterns.T @ F.softmax(BETA * patterns @ z, dim=0)


# compares two lists of images for similarity: each image is a tensor of tensors, each inner tensor is a row
# num_images is the number of images in each list, each image is size dim**2
# returns a list of floats, each is the fraction of pixels that are 'the same' for each pair of images with same indices
# 'the same' is determined by bound
def comp_images(images1, images2, num_images, dim, bound=DELTA):
    counts = []
    for i in range(num_images):
        count = 0
        rs = images1[i].numpy()
        ts = images2[i].numpy()
        for row_idx in range(len(rs)):
            for p_idx in range(len(rs[row_idx])):
                if abs(rs[row_idx][p_idx] - ts[row_idx][p_idx]) <= bound: count += 1
        counts.append(count / (dim ** 2))
    return counts


# given raw_vs and clustered_vs, choose n vectors however you want to become the vectors that are stored in the network
def sample_patterns(n, raw_vs, clustered_vs):
    # return torch.tensor(np.array([x for x in raw_vs]), dtype=float)[0:n]
    x = torch.tensor(np.array([x for x in raw_vs]), dtype=float)[0:n]
    return x


# generates NUM_GENERATED vectors clustered around c center vectors
# ^this cannot be number of stored vectors because for low number of stored, clusters cannot be filled
# ^^this must be larger than number of stored vectors because stored vectors are sampled from an array of this size
# center_threshold is how far center vectors are from each other
# cluster_threshold is how close vectors have to be to center vectors to be part of that cluster
# returns m vectors in a list, and then a c-long list of lists, each inner list contains a cluster of vectors
def generate_clusters(n, c, center_threshold, cluster_threshold, dim):
    # generate lots of initial vectors of size dim^2 (20 times number that I will eventually need)
    rand_vs = 255 * np.random.rand(100* ZETA * n, dim ** 2)

    # find center vectors that are far from each other
    center_vs = []
    for v in rand_vs:
        far = True
        for cv in center_vs:
            if normalize_dot(cv, v) > center_threshold:
                far = False
                break
        if far: center_vs.append(v)
        if len(center_vs) == c: break
    # if len(center_vs) != c:
    #     print(len(center_vs))
    #     raise Exception # check that there are c center vectors

    # group random vectors into clusters around centers
    # with this routine, the center vector itself may or may not be added to the cluster, and you're not guaranteed 20
    # full clusters
    clustered_vs = [[] for _ in range(c)]  # c-long array of arrays to hold clustered vectors
    raw_vs = []  # array to hold vectors added to a cluster
    num_gen = 0  # number of vectors added to a cluster
    for v in rand_vs:
        if num_gen < (ZETA * n):  # stop when NUM_)GENERATED vectors have been added
            # find what cluster it's closest to
            closest_idx = 0
            closest_dist = normalize_dot(center_vs[closest_idx], v)
            for idx, center_v in enumerate(center_vs):
                if normalize_dot(center_v, v) > closest_dist:
                    closest_idx = idx
                    closest_dist = normalize_dot(center_v, v)
            # add if it's close enough (closeness is above a threshold)
            if closest_dist > cluster_threshold:
                clustered_vs[closest_idx].append(v)
                raw_vs.append(v)
                num_gen += 1
    if num_gen != (ZETA * n):
        print(num_gen)
        raise Exception  # check that m vectors have been generated
    # for arr in clustered_vs:
    #     if not arr:
    #         print(arr)
    #         print(clustered_vs)
    #         raise Exception  # check that each cluster has at least 1 vector

    # print("cluster distribution:")
    # cluster_distr = []
    # for a in clustered_vs: cluster_distr.append(len(a))
    # print(cluster_distr)

    return raw_vs, clustered_vs


NUM_GENERATED_FRAC = 10
# given n stored vectors of size dim^2, returns an n-length array of floats, each float is the fraction of pixels that
# are 'correct' in the retrieval of the corresponding stored vector (see comp_images(...) for explanation of 'correct')
def test_storage(n, dim):
    # generation constants
    center_threshold = 0.74
    cluster_threshold = 0.75

    # generates NUM_GENERATED vectors of size dim^2 clustered around NUM_CENTERS center vectors
    # raw_vs, clustered_vs = generate_clusters(n // ETA, center_threshold, cluster_threshold, dim)


    raw_vs, clustered_vs = generate_clusters(n, 20, center_threshold, cluster_threshold, dim)

    # convert to tensor of n vectors, these will be stored; drawn from raw_vs, so in random clusters
    # print("len raw_vs " + str(len(raw_vs)))
    patterns = sample_patterns(n, raw_vs, clustered_vs)
    retrieved_arr = []
    truth_arr = []
    num_queries = 10  # how many queries are made
    if n < 10: num_queries = n

    for k in range(num_queries):
        # hide some of the vector to make a query
        query = np.concatenate([patterns[k][0: dim**2 // NU], np.ones(len(patterns[k]) - dim**2 // NU)])
        # retrieve a vector based on the query
        retrieved = np.reshape(update_rule(query, patterns, BETA), (dim, dim))  # reshapes it
        retrieved_arr.append(retrieved)
        # truth vector
        truth = np.reshape(patterns[k], (dim, dim))
        truth_arr.append(truth)

    # compare the retrieved array with the truth array and return the array of correctness
    return comp_images(retrieved_arr, truth_arr, num_queries, dim)


# given a min_n and max_n, computes a binary search for the maximum n that passes valid_storage on dim dimension
# returns the maximum valid n

VALID_BIAS = 0

def find_n(min_n, max_n, dim):

    # print("Finding maximum n....")

    first = min_n
    last = max_n

    while first <= last :
    # while (last - first > 10):
        midpoint = (first + last) // 2
        # print(first, last, midpoint)
        if valid_storage(midpoint, dim):
            first = (VALID_BIAS * first + midpoint) // (VALID_BIAS+1) + 1
            # print("worked")
        else:
            last = (midpoint + VALID_BIAS* last) // (VALID_BIAS + 1) - 1
            # print("didnt work")
    return (first+ last) // 2

NUM_TESTS = 10
VALID_CLOSE = 0.9
FRAC_CLOSE = 0.95

# given a dimension and n vectors stored, returns a boolean of whether the queried vector returns correctly
# ^ it does so by generating NUM_TESTS floats from test_storage, and requires all of the NUM_TESTS entries at maximum
# 1-FRAC_CLOSE entries to be greater than VALID_CLOSE, while all of the entries must be ones
def valid_storage(n, dim):

    # resultant array of size NUM_TESTS with the results of test_storage
    res = []

    # number of entries with value 1
    count_s = 0

    # number of entries with value >= VALID_CLOSE
    count_m = 0
    count = 0

    #
    while count < NUM_TESTS:
        # if len(res) % 250 ==0: print(len(res))
        x = test_storage(n, dim)
        # print(x)
        for i in x:
            # if i < VALID_CLOSE:
            #     return False
            # elif i == 1:
            #     count_s += 1
            # elif i >= VALID_CLOSE:
            #     count_m += 1
            count += 1
            if i != 1:
                count_m += 1
            if count_m > (1-FRAC_CLOSE) * NUM_TESTS:
                return False
            # res.append(i)

    # print(res)
    # print(count_s, count_m)
    # if count_s < NUM_TESTS * FRAC_CLOSE or count_s + count_m != NUM_TESTS:
    #     return False
    #if count_s
    return True

def main():
    # print(test_storage(1, 20))
    print(find_n(1, 2500, 30))
    res = {}
    # for i in range(32, 45):
    super_res = []
    # for i in range(5):
    #     test.append(find_n(10, 2000, 34))
    #     print(test)
    #
    # print(np.mean(test))
    curr_min = 1
    nmax = 2500

    for i in range(0, 5):
        for i in range(25, 38):
            # x = find_n(curr_min,10000000,i)
            x = find_n(curr_min, nmax, i)
            res[i] = x
            # curr_min = x
            print(i, x)
            if x == nmax - 1:
                break
        print(res)
        super_res.append(res)
    print(super_res)


if __name__ == "__main__":
    main()
