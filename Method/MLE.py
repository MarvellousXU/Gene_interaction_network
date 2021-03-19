
import numpy as np
import Clustering_metric as cm
from sklearn.preprocessing import normalize
class MLE(object):
    def __init__(self,A,K,Labels):
        self.A = A  # m * n * n tensor
        self.A_copy = np.copy(A)
        self.n = self.A.shape[1] # the number of nodes
        self.m = self.A.shape[0] # the number of layers
        self.K = K # the number of communities needed to specify
        self.True_labels = Labels # the community lables of nodes

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
            Mat = normalize(np.diag(np.ones(self.K)) + 0.8,'l1')
            self.b = Mat[km.labels_]
        except:
            self.b = normalize(np.random.uniform(0, 1, [self.n, self.K]), norm='l1')
        self.P = np.zeros([self.m, self.K, self.K])
        self.Pi = self.b.mean(axis=0) # the prior of each communities
        self.Best_labels = []
    def Iter(self):
        for i in range(self.K):
            for j in np.arange(i,self.K):
                Tau_ij = np.matmul(self.b[:, i].reshape(1, -1).T, self.b[:, j].reshape(1, -1))
                Tau_ij_d = Tau_ij - np.diag(Tau_ij.diagonal())  # remove diagonal value
                Temp_mu_ij = (self.A * Tau_ij_d).sum(axis=(1, 2)) / Tau_ij_d.sum()
                self.P[:,i,j],self.P[:,j,i] = Temp_mu_ij, Temp_mu_ij
        for i in range(self.n):
            Mat_i = np.delete(self.A[:,i,:],i,axis=1).T
            proport = np.log(self.Pi)
            for k in range(self.K):
                for l in range(self.K):
                    temp_prob = self.P[:,k,l]
                    Main_kl = temp_prob ** Mat_i * (1-temp_prob) ** (1-Mat_i)
                    b_i = np.delete(self.b[:, l], i)
                    proport[k] += np.log(Main_kl.T ** b_i).sum()
            Temp_prob_log_1 = np.array(proport) - max(proport)
            Temp_prob_log_2 = [max(-10,i) for i in Temp_prob_log_1]
            Temp_b = np.exp(Temp_prob_log_2)/sum(np.exp(Temp_prob_log_2)) * self.Pi
            self.b[i] = Temp_b/sum(Temp_b)
        self.Pi = self.b.mean(axis=0) # the prior of each communities

    def Get_labels(self):
        Labels = []
        for i in range(self.n):
            temp = self.b[i].tolist()
            index = temp.index(max(temp))
            Labels.append(index)
        return(np.array(Labels))
    def Get_error(self):
        error = cm.Clus_dist(np.array(self.Get_labels()),self.True_labels)
        return(error)
    def Training(self):
        Labels_set = [self.Get_labels()]
        Stop = False
        while Stop==False:
            self.Iter()
            Labels_set.append(self.Get_labels())
            Stop = (sum(Labels_set[-1]==Labels_set[-2])==self.n)
            if self.Get_error()==0:
                Stop = True
        return(self.Get_error())
    def Optimal_thres(self):
        quantile = np.array([0.2 + 0.1 * i for i in range(7)])
        Cut = np.quantile(self.A, quantile)
        Error_set = []
        Labels_set = []
        for cut in Cut:
            self.A = np.array(self.A_copy > cut, dtype=int)
            self.Init()
            Error_temp = self.Training()
            Error_set.append(Error_temp)
            Labels_set.append(self.Get_labels())
        self.Best_labels = Labels_set[Error_set.index(min(Error_set))]
        return(min(Error_set))





