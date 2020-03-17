from numba import jit, prange
import numpy as np


@jit(nopython=True, parallel=True)
def calculate_gram_mat(train_set, length, gram_mat):
    for i in prange(length):
        if i % 50 == 0:
            print(i)
        for j in range(length):
            if i != j:
                gram_mat[i][j] = np.float32(np.dot(train_set[i].ravel(),
                                                   train_set[j].ravel()))
            else:
                gram_mat[i][j] = 1

    return gram_mat


def calculate_and_save_gram_mat(train_set, length, csv_file_name):
    gram_mat_init = np.empty([length, length], dtype=np.float32)

    gram_mat = calculate_gram_mat(train_set, length, gram_mat_init)
    gram_mat = flatten_gram_mat(gram_mat, length)

    np.savetxt(fname=csv_file_name, X=gram_mat, delimiter=',')
    return gram_mat


@jit(nopython=True, parallel=True)
def flatten_gram_mat(gram_mat, length):
    print("Flattening Gram Matrix")
    max_col_val = np.zeros(length, dtype=np.float32)

    for j in prange(length):
        max_col_val[j] = 0
        for i in range(length):
            if i != j and max_col_val[j] < gram_mat[i][j]:
                max_col_val[j] = gram_mat[i][j]

    for j in prange(length):
        for i in range(length):
            if i != j:
                gram_mat[i][j] = gram_mat[i][j] / max_col_val[j]

    return gram_mat


'''
def linear_ker(x, z):
    return np.dot(x, z)


def polynomial_ker(x, z, dim=2):
    return (np.dot(x, z) + 1) ** dim


def rbf_ker(x, z, sigma=5.12):
    return np.exp(-(np.linalg.norm(x - z) ** 2) / (2.0 * (sigma ** 2)))
'''
