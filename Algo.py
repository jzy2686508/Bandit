import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import scipy.stats
from Bandit_env.Bandit_env import Bandit
import pandas as pd

torch.cuda.set_device(3)
total_R = 300000
ite = 1
sep_R = total_R // ite

results = np.zeros(total_R)
myparams = {'N': 25, 'T': 1, 'R': sep_R, 'mean_reward': [0, 0], 'var_reward': [1, 1],
            'clip': 0.1, 'algo': 'thompson'}

start = time.perf_counter()
for i in range(ite):
    mbit = Bandit(params=myparams, cuda_available=True)
    est = mbit.regular_est().cpu()
    results[range(sep_R * i, (i+1) * sep_R)] = est
end = time.perf_counter()
print(end-start)

# save figure
str_label = 'AW-AIPW  ' + 'N='+ str(myparams['N']) + ' T=' + str(myparams['T']) + \
            ' algo=' + str(myparams['algo']) + ' clip=' + str(myparams['clip'])
plt.hist(results,density=True, bins=200)
x_temp = np.arange(-4,4,0.02)
y_temp = scipy.stats.norm.pdf(x_temp)
plt.plot(x_temp,y_temp,label='Standard Normal')
plt.title(str_label)
plt.legend()
plt.show()
critical_value = scipy.stats.norm.ppf(q=1-0.05/2)
# critical_value = scipy.stats.t.ppf(q=1-0.05/2,df=myparams['N']-2)
print("Type-1 error rate", (results.__abs__() > critical_value).sum()/ results.size)


