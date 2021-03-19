import numpy as np
import sys
from sklearn.preprocessing import normalize
import Clustering_metric as Cm
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
sys.path.append('Simulation/')
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
def Mean_init(K, m):
    # initialize the mean vectors for K communities
    List = {}
    for i in range(K):
        for j in np.arange(i, K):
            Temp_ij = np.random.uniform(-1, 1, m)
            if i != j:
                List.update({(i, j): Temp_ij})
                List.update({(j, i): Temp_ij})
            else:
                List.update({(i, j): Temp_ij})
    return (List)
def Vari_init(K, m):
    List = {}
    for i in range(K):
        for j in np.arange(i, K):
            Temp_ij = np.eye(m)
            if i != j:
                List.update({(i, j): Temp_ij})
                List.update({(j, i): Temp_ij})
            else:
                List.update({(i, j): Temp_ij})
    return (List)
def Pair_not_diag(n):
    Set = []
    for i in range(n):
        for j in range(n):
            if i != j:
                Set.append([i, j])
    return (Set)
def Max_step(X):
    X = np.array(X)
    X[X<0.001] = 0.01
    X = X/sum(X)
    return(X)

class VEM(object):
    def __init__(self, A,K,Labels):
        self.A = A  # m * n * n tensor
        self.n = self.A.shape[1] # the number of nodes
        self.m = self.A.shape[0] # the number of layers
        self.K = K # the number of communities needed to specify
        self.True_labels = Labels # the community lables of nodes
        self.Error = False # whether the program has a error in community detection
    def Init(self):
        # randomly initialization
        try:
            ALL_eigh = []
            for i in range(self.A.shape[0]):
                (sigmai, Ui) = np.linalg.eigh(self.A[i, :, :])
                ind_1 = np.argsort(-abs(sigmai))
                eigen_vectori = Ui[:, ind_1[0:self.K]]
                ALL_eigh.append(eigen_vectori)
            Combine = np.concatenate(ALL_eigh, axis=1)
            km = KMeans(n_clusters=self.K).fit(Combine)
            Mat = normalize(np.diag(np.ones(self.K)),'l1')
            self.Tau = Mat[km.labels_]
        except:
            self.Tau = normalize(np.random.uniform(0, 1, [self.n, self.K]), norm='l1')
        self.Mean_list = Mean_init(self.K, self.m) # the dictionary of mean vector for community-to-community
        self.Vari_list = Vari_init(self.K, self.m) # the dictionary of covariance matrix for community-to-community
        self.alpha = self.Tau.mean(axis=0) # the prior of each communities
    def Iter(self):
        for i in range(self.K):
            for j in np.arange(i, self.K):
                Tau_ij = np.matmul(self.Tau[:, i].reshape(1, -1).T, self.Tau[:, j].reshape(1, -1))
                Tau_ij_d = Tau_ij - np.diag(Tau_ij.diagonal())  # remove diagonal value
                Temp_mu_ij = (self.A * Tau_ij_d).sum(axis=(1, 2)) / Tau_ij_d.sum()
                # estimate covariance matrix
                Temp_mu_ij_sqr = (self.A - Temp_mu_ij.reshape(self.m, 1, 1)) * np.sqrt(Tau_ij_d)
                Set_for_vari = Pair_not_diag(self.n)
                A_ij_list = np.array([Temp_mu_ij_sqr[:, index[0], index[1]] for index in Set_for_vari])
                Var_mat_ij = np.matmul(A_ij_list.T, A_ij_list) / Tau_ij_d.sum() # estimated variance matrix
                if i != j:
                    self.Mean_list.update({(i, j): Temp_mu_ij})
                    self.Mean_list.update({(j, i): Temp_mu_ij})
                    self.Vari_list.update({(i, j): Var_mat_ij})
                    self.Vari_list.update({(j, i): Var_mat_ij})
                else:
                    self.Mean_list.update({(i, j): Temp_mu_ij})
                    self.Vari_list.update({(i, j): Var_mat_ij})
        # updating tau
        for i in range(self.n):
            Temp_prob_log = []
            Set_K = []
            Tau_temp_i = np.delete(self.Tau,i,axis=0)
            for k in range(self.K):
                Temp_mat = np.array([multivariate_normal.pdf(self.A[:, i, :].T, self.Mean_list.get((k, l)), self.Vari_list.get((k, l))) for l in range(self.K)]).T
                Tm_k = np.delete(Temp_mat,i,axis=0)
                Set_K.append(Tm_k**Tau_temp_i)
            for j in np.arange(0,self.K):
                temp = Set_K[j].copy()
                #Temp_prob_log.append((np.log(temp+10**-300)).sum())
                Temp_prob_log.append((np.log(temp)).sum())
            Temp_prob_log_1 = np.array(Temp_prob_log) - np.max(Temp_prob_log)#-------------------------
            #Temp_prob_log_1[Temp_prob_log_1<=-20.] = -20.
            Temp_prob_log_2 = np.exp(Temp_prob_log_1) * self.alpha
            self.Tau[i] = Temp_prob_log_2/sum(Temp_prob_log_2)
        # updating alpha
        self.alpha = self.Tau.sum(axis=0)/self.n
        if min(self.alpha)<0.01:
            print('You may consider a smaller K: the clusters overlap')
    def Get_labels(self):
        Labels = []
        for i in range(self.n):
            temp = self.Tau[i].tolist()
            index = temp.index(max(temp))
            Labels.append(index)
        return(np.array(Labels))
    def Get_error(self):
        error = Cm.Clus_dist(np.array(self.Get_labels()),self.True_labels)
        return(error)

    def Training(self):
        self.Init()
        Labels_set = [self.Get_labels()]
        Stop = False
        L = 0
        while Stop==False:
            self.Iter()
            L += 1
            Labels_set.append(self.Get_labels())
            Stop = (sum(Labels_set[-1]==Labels_set[-2])==self.n)
            if (self.Get_error()==0) or (L==50):
                Stop = True
        return(self.Get_error())
    def Fix_step_training(self,step = 100):
        self.Init()
        for i in range(step):
            self.Iter()
            L_CSBM = self.Get_labels() + 1
            L_CSBM_p = Cm.Give_true_labels(self.True_labels, L_CSBM)
            print(sum(L_CSBM_p != self.True_labels))





