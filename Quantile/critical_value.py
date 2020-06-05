import numpy as np
import scipy.stats
import torch
import torch.distributions


quantile = np.zeros(shape=(100,100,10))
def compute_t_critical(N,T):
    a = np.random.standard_t(df=N-2,size=(1000000,T))
    a = a.sum(axis=1)/ np.sqrt(T)
    a.sort()
    return np.array([a[10000], a[25000], a[50000], a[100000], a[500000], a[900000], a[950000], a[975000], a[990000],
                     (-a[25000]+a[975000])/2])

for i in range(30,100):
    print(i)
    for j in range(3,30):
        quantile[i,j,:] = compute_t_critical(i,j)
    np.save('quantile_standardized_t/quantile.npy', quantile)

