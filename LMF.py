import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import Clustering_metric as cm
rpy2.robjects.numpy2ri.activate()
robjects.r.source('F:\\PythonProject\\Network Data\\Method\\LMF.r')
from rpy2.robjects.packages import importr
kernlab = importr('kernlab')
clue = importr('clue')
psych = importr('psych')
igraph = importr('igraph')
def LMF(A,K,True_label):
    Cut = np.array([0.2 +0.1*i for i in range(7)])
    Cut = np.quantile(A,Cut)
    Error_set = []
    List_set = []
    for cut in Cut:
        temp_A = np.array(A>cut,dtype=int)
        temp_A_input = [temp_A[i,:,:] for i in range(A.shape[0])]
        psi_hat = robjects.r.lmfo(temp_A_input,A.shape[1], K)
        List1 = np.array(psi_hat)
        List_set.append(List1)
        List2 = True_label
        Error_set.append(cm.Clus_dist(List1, List2))
    return(min(Error_set),List_set[Error_set.index(min(Error_set))])
