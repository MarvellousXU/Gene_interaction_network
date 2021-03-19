# For Lei, Chen, Lynch 2019 Consistent community detection in multi-layer network data.

rm(list=ls())

source('NetTensor.r')

n <- 200
k <- 3 # number of communities
m <- 3 # layers 


Btensor <- array(0, dim = c(3, 3, 3)) # B tensor should be m*k*k

# The following example is corresponding to the motivating example eq (1) in the paper. 
Btensor[1, ,] <- matrix(c(0.6,0.4,0.4,0.4,0.2,0.2,0.4,0.2,0.2), 3, 3)
Btensor[2, ,] <- matrix(c(0.2,0.4,0.2,0.4,0.6,0.4,0.2,0.4,0.2), 3, 3)
Btensor[3, ,] <- matrix(c(0.2,0.2,0.4,0.2,0.2,0.4,0.4,0.4,0.6), 3, 3)
       
clust.size <- c(rep(floor(n/k), k-1),n-(k-1)*floor(n/k)) # equal cluster size
Theta <- Generate.theta(n, k, clust.size)
A <- Generate.data(Theta, Btensor, self = 0)

res <- GetCluster(A, k)$id  # use all layers
res.2 <- GetCluster.1(A[1,,], k)$id  #use the first layer only 
res.3 <- GetCluster.1(A[1,,], 2)$id  # the first layer only show two distinct clusters
               
