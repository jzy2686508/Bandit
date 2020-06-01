import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import scipy.stats
from Bandit_env.Bandit_env import Bandit

total_R = 100000
ite = 1
sep_R = total_R // ite

results = np.zeros(total_R)
myparams = {'N': 25, 'T': 25, 'R': sep_R, 'mean_reward': [0, 0], 'var_reward': [1, 1],
            'clip': 0.1, 'algo': 'greedy'}

start = time.perf_counter()
for i in range(ite):
    mbit = Bandit(params=myparams, cuda_available=True)
    est = mbit.aw_aipw().cpu()
    results[range(sep_R * i, (i+1) * sep_R)] = est
end = time.perf_counter()
print(end-start)

plt.hist(results,density=True, bins=100)
x_temp = np.arange(-4,4,0.02)
y_temp = scipy.stats.norm.pdf(x_temp)
plt.plot(x_temp,y_temp)
plt.show()
critical_value = scipy.stats.norm.ppf(q=1-0.05/2)
critical_value = scipy.stats.t.ppf(q=1-0.05/2,df=myparams['N']-2)
print("Type-1 error rate", (results.__abs__() > critical_value).sum()/ results.size)



plt.hist(mbit.bols[:,0,0].cpu().numpy(),density=True,bins=100)
x_temp = np.arange(-4,4,0.02)
y_temp = scipy.stats.norm.pdf(x_temp)
plt.plot(x_temp,y_temp)
plt.show()