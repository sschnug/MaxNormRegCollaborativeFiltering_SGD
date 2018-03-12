import math
import itertools
import pickle
import numpy as np
np.random.seed(1)

""" IO
    --
"""
def read_ml(fp, n, delimiter='\t'):
    """ Slow but safe (customized processing of rating -> int!)
        Warning: some movielens-datasets have a header! """
    data_um = np.empty((n, 2), dtype=np.uint32)
    data_r = np.empty(n, dtype=np.float32)

    f = open(fp, 'r')
    for ind, line in enumerate(f):  # header
        line_data = line.split(delimiter)

        data_um[ind, 0] = int(line_data[0])
        data_um[ind, 1] = int(line_data[1])
        data_r[ind] = float(line_data[2])

    return data_um, data_r

""" Preprocessing """
def map_gapless(mmap):
    unique_user_ids = np.unique(mmap[:, 0])
    unique_movie_ids = np.unique(mmap[:, 1])

    user_id_2_user_0nid = {}
    movie_id_2_movie_0nid = {}
    user_0nid_inverse = {}
    movie_0nid_inverse = {}

    user_count = itertools.count()
    movie_count = itertools.count()

    for i in unique_user_ids:
        new_id = next(user_count)
        user_id_2_user_0nid[i] = new_id
        user_0nid_inverse[new_id] = i

    for i in unique_movie_ids:
        new_id = next(movie_count)
        movie_id_2_movie_0nid[i] = new_id
        movie_0nid_inverse[new_id] = i

    with open('user_id_2_user_0nid.pickle', 'wb') as f:
        pickle.dump(user_id_2_user_0nid, f, pickle.HIGHEST_PROTOCOL)

    with open('movie_id_2_movie_0nid.pickle', 'wb') as f:
        pickle.dump(movie_id_2_movie_0nid, f, pickle.HIGHEST_PROTOCOL)

    with open('user_0nid_inverse.pickle', 'wb') as f:
        pickle.dump(user_0nid_inverse, f, pickle.HIGHEST_PROTOCOL)

    with open('movie_0nid_inverse.pickle', 'wb') as f:
        pickle.dump(movie_0nid_inverse, f, pickle.HIGHEST_PROTOCOL)

    return user_id_2_user_0nid, movie_id_2_movie_0nid, user_0nid_inverse, movie_0nid_inverse

""" MAX NORM OPT """
class SGD(object):
    """ Stochastic Gradient (Online) """
    def __init__(self, n, m, k, b):
        """ """
        self.n = n
        self.m = m
        self.k = k  # dimension of latent factors
        self.b = b  # regulization value / approximated rank

        # define L/R (latent vectors) and assign a random starting-point
        self.L = np.random.randn(n, k) * 0.01
        self.R = np.random.randn(m, k) * 0.01

    def calc_train_error(self, ids, ratings, clip_lb, clip_ub):
        n = ids.shape[0]
        e = 0.0
        for ind in range(n):
            i, j = ids[ind]
            r = ratings[ind]
            e += math.pow(
                np.clip(np.dot(self.L[i, :], self.R[j, :].T), clip_lb, clip_ub) - r, 2)  # clip vs. no-clip
        return math.pow(e / n, 0.5)

    def train(self, ids, ratings, test_ids, test_ratings, epochs=20, gamma=.05, gamma_red_factor=0.8,
              show_progress=True, show_train_error=True):

        def project(v):
            v_norm = np.linalg.norm(v)
            if np.square(v_norm) >= self.b:
                return (math.sqrt(self.b) * v) / v_norm
            else:
                return v

        train_lb, train_ub = np.amin(ratings), np.amax(ratings)

        n_obs = ids.shape[0]
        perm_indices = np.arange(n_obs)

        for epoch in range(epochs):
            if show_progress:
                print('epoch: ', epoch, '/', epochs)
            if show_train_error:
                if epoch != 0:
                    if (epoch - 1) % 2 == 0:
                        print(' calc train-error: ')
                        e = self.calc_train_error(ids, ratings, train_lb, train_ub)
                        print(' -> ', e)

                        print(' calc test-error: ')
                        e = self.calc_train_error(test_ids, test_ratings, train_lb, train_ub)
                        print(' -> ', e)

            np.random.shuffle(perm_indices)
            for ind in perm_indices:
                i, j = ids[ind]
                r = ratings[ind]

                pred = np.dot(self.L[i, :], self.R[j, :].T)

                # update rule
                grad = pred - r
                L_pre = self.L[i, :] - gamma * grad * self.R[j, :]
                R_pre = self.R[j, :] - gamma * grad * self.L[i, :]

                # projection
                self.L[i, :] = project(L_pre)
                self.R[j, :] = project(R_pre)

            # decr learning-rate
            gamma *= gamma_red_factor

            # if epoch % 5 == 0:
            #     np.savez_compressed('DEBUG_LR_epoch_b200_k30' + str(epoch), L=self.L, R=self.R)

""" TEST """

# Read
# data_um, data_r = read_mlk('ml-100k\\u.data', 100000)
# data_um, data_r = read_ml('Y:\\ml-10m\\ml-10M100K\\ratings.dat', 10000054, delimiter='::')
data_um, data_r = read_ml('Y:\\ml-1m\\ml-1m\\ratings.dat', 1000209, delimiter='::')

# Preprocess
user_id_2_user_0nid, movie_id_2_movie_0nid, user_0nid_inverse, movie_0nid_inverse = map_gapless(data_um)
for ind, i in enumerate(data_um):
    data_um[ind] = [user_id_2_user_0nid[i[0]], movie_id_2_movie_0nid[i[1]]]

perm = np.arange(data_um.shape[0])
np.random.shuffle(perm)
train_inds, test_inds = perm[:800000], perm[800000:]   # TRAIN TEST SPLIT

test_um, test_r = np.copy(data_um[test_inds]), np.copy(data_r[test_inds])
train_um, train_r = np.copy(data_um[train_inds]), np.copy(data_r[train_inds])
train_r_mean = np.mean(train_r)
train_r -= train_r_mean
test_r -= train_r_mean
unique_users, unique_movies = np.unique(data_um[:, 0]).shape[0], np.unique(data_um[:, 1]).shape[0],

print('train size')
print(train_um.shape)
print(train_r.shape)

print('test size')
print(test_um.shape)
print(test_r.shape)

# Learn
mf = SGD(unique_users, unique_movies, k=50, b=2.1)
mf.train(train_um, train_r, test_um, test_r, epochs=50, gamma=.1, gamma_red_factor=0.95,
         show_progress=True, show_train_error=True)
