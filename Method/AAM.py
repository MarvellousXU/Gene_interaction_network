# The method on averaged adjacency matrix
from sklearn.cluster import KMeans
import numpy as np
from functools import reduce
import Clustering_metric as cm
def AAM_new(A,K,True_label):
    Error_set = []
    mean_A = A.mean(axis=0)
    LL = np.sqrt(mean_A.mean(axis=0))
    D_sqr = []
    for i in LL:
        if i == 0:
            D_sqr.append(0)
        else:
            D_sqr.append(1 / i)
    D_sqr = np.diag(D_sqr)
    L = np.eye(A.shape[1]) - reduce(np.dot, [D_sqr, mean_A, D_sqr])
    (sigma, U) = np.linalg.eig(L)
    ind = np.argsort(sigma)
    eig_vector = U[:, ind[0:K]]
    kmeans = KMeans(n_clusters=K).fit(eig_vector)
    List1 = np.array(kmeans.labels_)
    List2 = True_label
    Error_set.append(cm.Clus_dist(List1, List2))
    return(min(Error_set))

def AAM(A,K,True_label):
    Cut = np.array([0.2 +0.1*i for i in range(7)])
    Cut = np.quantile(A,Cut)
    Error_set = []
    Label_set = []
    for cut in Cut:
        temp_A = np.array(A>cut,dtype=int)
        mean_A = temp_A.mean(axis=0)
        D = np.diag(np.sum(mean_A, axis=0))
        LL = np.sqrt(mean_A.mean(axis=0))
        D_sqr = []
        for i in LL:
            if i==0:
                D_sqr.append(0)
            else:
                D_sqr.append(1/i)
        D_sqr = np.diag(D_sqr)
        L = np.eye(A.shape[1]) - reduce(np.dot, [D_sqr, mean_A, D_sqr])
        (sigma, U) = np.linalg.eig(L)
        ind = np.argsort(sigma.real)
        eig_vector = U[:, ind[0:K]]
        kmeans = KMeans(n_clusters=K).fit(eig_vector.real)
        Label_set.append(kmeans.labels_)
        List1 = np.array(kmeans.labels_)
        List2 = True_label
        Error_set.append(cm.Clus_dist(List1, List2))
    return(min(Error_set),Label_set[Error_set.index(min(Error_set))])
