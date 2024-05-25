'''FUNCTIONS'''

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform, euclidean
import matplotlib.pyplot as plt



def Kernel_matrix(df, epsilon):
    """Compute a kernel matrix with adaptive bandwidth
    df = data frame on which to compute the kernel matrix
    epsilon = number of neighbors to consider in the bandwidth calculation"""

    D = squareform(pdist(df, 'euclidean')) # euclidian distance matrix
    dist_sort = np.sort(D, axis=1)
    sigmas = dist_sort[:, round(epsilon)]
    Sig = np.outer(sigmas, sigmas)  # the adaptive bandwidth
    kernel_matrix = np.exp(-(D ** 2) / Sig)
    return kernel_matrix


def LG_RW(W, k=None, initial=None):
    """ compute the Laplacian graph (type=random walk): L = D^(-1)K"""
    D = np.diag(np.power(np.sum(W, axis=1), -1)) # inverse of the degree matrix
    Lrw = D @ W # the random walk laplacian

    v, d, _ = np.linalg.svd(Lrw) # calculate its singular value decomposition
    idx_ = np.argsort(d)[::-1] # sort by the singular values
    v = v[:, idx_]
    if k:
        v = v[:, :k]
        d = d[:k]
    return Lrw, d, v


def LG_K(W, k=None, initial=None):
    """ compute Laplacian graph (type=regular): L = D - K"""
    D = np.diag(np.sum(W, axis=1)) # the degree matrix
    Lrw = D - W # the laplacian matrix

    v, d, _ = np.linalg.svd(Lrw) # calculate its singular value decomposition
    idx_ = np.argsort(d)[::-1] # sort by the singular values
    v = v[:, idx_]
    if k:
        v = v[:, :k]
        d = d[:k]
    return Lrw, d, v


def LG_sym(W, k=None, initial=None):
    """ compute Laplacian graph (type=Symmetric): L = D^(-0.5)K D^(-0.5)"""
    D = np.diag(np.sum(W, axis=1) ** (-0.5)) # D ** (-0.5), D = degree matrix
    Lrw = D @ W @ D # the operator of the symmetric matrix

    d, v = np.linalg.eigh(Lrw) # calculate its eigen decomposition
    idx_ = np.argsort(d)[::-1] # sort by the eigen values
    v = v[:, idx_]
    if k != None:
        v = v[:, :k]
        d = d[:k]
    return Lrw, d, v


def circ_convolution(x, y):
    x_ext = np.concatenate((x, x))
    a = np.correlate(x_ext, y, )  # ,mode = 'same'
    return a


def calc_differential_vec(L_A, v_B, k, Q=None):
    """ Calculate differential vectors, as describes in Alg 3.1.
    L_A = the Laplacian matrix we filter.
    v_B = the eigenvectors of the second modality we filter L_A with.
    K = the number of leading eigenvectors we use in the filter."""
    U1 = v_B[:, :k]
    Q1 = U1 @ U1.T
    Q1 = np.eye(Q1.shape[0]) - Q1
    Q1 = Q1 @ L_A @ Q1

    s, u1 = np.linalg.eigh(Q1)
    idx_order = np.argsort(s)[::-1]
    u1 = u1[:, idx_order]
    if Q:
        return Q1, s, u1
    else:
        return s, u1



def calc_sig_to_noise(x, y, s, display=True, sort=True):
    """Function to compute the SNR ratio.
    x is the estimation of the latent variable.
    y is the ground truth latent variable.
    display: whether to display the SNR visualization or not.
    sort: whether to sort x (the estimation) according to y (the ground truth)"""
    if sort:
        sig = pd.Series(x[np.argsort(y)])
    else:
        sig = pd.Series(x)

    amp = np.mean(np.abs(sig.rolling(window=s).mean()))  # - sig.rolling(window=s).mean().mean()
    print(sig.rolling(window=s).mean().mean())
    noise = np.mean(np.power(sig.rolling(window=s).std(), 2))
    sig_noise = amp / noise

    vecs = np.abs(sig.rolling(window=s).mean() - sig.rolling(window=s).mean().mean()) / np.abs(
        sig.rolling(window=s).std())
    p_val = len(np.where(vecs <= 1)[0]) / len(vecs)

    if display:
        # calculate a 60 day rolling mean and plot
        sig.rolling(window=s).mean().plot(style='k')
        # add the 20 day rolling standard deviation:
        sig.rolling(window=s).std().plot(style='b')
        plt.show()
        print("signal to noise: " + str(sig_noise))
        plt.hist(vecs, bins=100)
        plt.show()
        print("P value of signal to noise: " + str(p_val))

    return sig_noise



