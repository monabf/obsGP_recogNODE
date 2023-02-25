import time

import torch
from pykeops.torch import LazyTensor

from utils.pytorch_utils import StandardScaler

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


# Different classes of useful objects using Keops, which is long to compile
# so not open if not necessary

# k-Means clustering with KeOps
# https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html
# https://en.wikipedia.org/wiki/K-means_clustering
def kMeans(x, k=10, Niter=100, dist='Euclidean', init='Forgy', verbose=False):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    x = x.contiguous()
    scaler = StandardScaler(x)
    x = scaler.transform(x)  # Standardize dataset first
    N, D = x.shape  # Number of samples, dimension of the ambient space
    if init == 'Forgy':
        random_idx = torch.randperm(N)[:k]
        c = x[random_idx].clone()  # Random initialization for the centroids
    elif init == 'kMeans++':
        # https://en.wikipedia.org/wiki/K-means%2B%2B
        raise NotImplementedError
    elif init == 'BradleyFayyad':
        # https://www.semanticscholar.org/paper/Refining-Initial-Points-for-K-Means-Clustering-Bradley-Fayyad/3b4b6b3b6f13f2de00a213a822e7c005e034adae
        raise NotImplementedError
    elif init == 'First':
        c = x[:k, :].clone()  # Simplistic initialization for the centroids
    else:
        raise NotImplementedError
    if dist == 'cosine':
        # Normalize the centroids for the cosine similarity:
        c = torch.nn.functional.normalize(c, dim=1, p=2)
    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, k, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        if dist == 'Euclidean':
            D_ij = ((x_i - c_j) ** 2).sum(
                -1)  # (N, K) symbolic squared distances
            cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster
        elif dist == 'cosine':
            S_ij = x_i | c_j  # (N, K) symbolic Gram matrix of dot products
            cl = S_ij.argmax(dim=1).long().view(-1)  # Points -> Nearest cluster
        else:
            raise NotImplementedError

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        if dist == 'Euclidean':
            # Divide by the number of points per cluster:
            Ncl = torch.bincount(cl, minlength=k).type_as(c).view(k, 1)
            c /= Ncl  # in-place division to compute the average
        elif dist == 'cosine':
            # Normalize the centroids, in place:
            c[:] = torch.nn.functional.normalize(c, dim=1, p=2)
        else:
            raise NotImplementedError

    if verbose:  # Fancy display -----------------------------------------------
        if 'cuda' in str(x.device):
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the {dist} metric with {N:,} points in dimension "
            f"{D:,}, K = {k:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, scaler.inverse_transform(c)
