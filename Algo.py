import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import scipy.stats
from Bandit_env.Bandit_env import Bandit

total_R = 100000
ite = 5
sep_R = total_R // ite

start = time.perf_counter()
results = np.zeros(total_R)
myparams = {'N': 100, 'T': 25, 'R': sep_R, 'mean_reward': [0, 0], 'var_reward': [1, 1],
            'clip': 0.1, 'algo': 'greedy'}
for i in range(ite):
    mbit = Bandit(params=myparams, cuda_available=True)
    mbit.regular_est()
    results[range(sep_R * i, (i+1) * sep_R)] = mbit.diff
end = time.perf_counter()
print(end-start)

plt.hist(results,density=True, bins=80)
x_temp = np.arange(-4,4,0.02)
y_temp = scipy.stats.norm.pdf(x_temp)
plt.plot(x_temp,y_temp)
plt.show()
print("Type-1 error rate", (results.__abs__() > scipy.stats.norm.ppf(q=1-0.05/2)).sum() / results.size)
