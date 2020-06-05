import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import scipy.stats
from Bandit_env.Bandit_env import Bandit
import pandas as pd

#simN: 3 - 40

#simT: 3 - 40
torch.cuda.set_device(2)

quantile_point = np.array([0.01,0.025,0.05,0.1,0.5,0.9,0.95,0.975,0.99])
quantile_normal = scipy.stats.norm.ppf(q=quantile_point)
x_temp = np.arange(-4,4,0.02)
y_temp = scipy.stats.norm.pdf(x_temp)

total_R = 200000
ite = 1
sep_R = total_R // ite
results = np.zeros(total_R)
RESULT = np.zeros(shape=(61,61,10))
RESULT = np.load('Quantile/empirical_quantile_normal/BOLS_greedy.npy')
# N;3-41
# T:3-41

for N in range(3,41):
    print(N)
    start = time.perf_counter()
    for T in range(3,41):
        myparams = {'N': N, 'T': T, 'R': sep_R, 'mean_reward': [0, 0], 'var_reward': [1, 1],
                    'clip': 0.1, 'algo': 'greedy'}
        for i in range(ite):
            mbit = Bandit(params=myparams, cuda_available=True)
            est = mbit.aw_aipw().cpu()
            results[range(sep_R * i, (i + 1) * sep_R)] = est
        for i in range(9):
            RESULT[N,T,i] = (results < quantile_normal[i]).mean()
        RESULT[N,T,9] = RESULT[N,T,1] + 1 - RESULT[N,T,7]
        # Graph
        # str_label = 'AW-AIPW  ' + 'N=' + str(myparams['N']) + ' T=' + str(myparams['T']) + \
        #             ' algo=' + str(myparams['algo']) + ' clip=' + str(myparams['clip'])
        # plt.hist(results, density=True, bins=200)
        # plt.plot(x_temp, y_temp, label='Standard Normal')
        # plt.title(str_label)
        # plt.legend()
        # plt.savefig('Figure/AW-AIPW/'+str_label+'.pdf')
        # plt.close()
    end = time.perf_counter()
    print(end - start)
    np.save('RESULT/Thompson/AW_THOMPSON.npy', RESULT)


# simulate emprical critical point
for N in range(3,41):
    print(N)
    start = time.perf_counter()
    for T in range(3,41):
        myparams = {'N': N, 'T': T, 'R': sep_R, 'mean_reward': [0, 0], 'var_reward': [1, 1],
                    'clip': 0.1, 'algo': 'thompson', 'rwdtype': 'normal'}
        for i in range(ite):
            mbit = Bandit(params=myparams, cuda_available=True)
            est = mbit.batched_ols().cpu()
            results[range(sep_R * i, (i + 1) * sep_R)] = est
        results.sort()
        RESULT[N, T, :9] = results[(total_R*quantile_point).astype(int)]
        RESULT[N, T, 9] = (RESULT[N,T,7] - RESULT[N,T,1])/2
    end = time.perf_counter()
    print(end - start)
    np.save('Quantile/empirical_quantile_normal/BOLS_thompson.npy', RESULT)

    # simulate emprical critical point
    for N in range(3, 41):
        print(N)
        start = time.perf_counter()
        for T in range(3, 41):
            myparams = {'N': N, 'T': T, 'R': sep_R, 'mean_reward': [0, 0], 'var_reward': [1, 1],
                        'clip': 0.1, 'algo': 'thompson', 'rwdtype': 'normal'}
            for i in range(ite):
                mbit = Bandit(params=myparams, cuda_available=True)
                est = mbit.aw_aipw().cpu()
                results[range(sep_R * i, (i + 1) * sep_R)] = est
            results.sort()
            RESULT[N, T, :9] = results[(total_R * quantile_point).astype(int)]
            RESULT[N, T, 9] = (RESULT[N, T, 7] - RESULT[N, T, 1]) / 2
        end = time.perf_counter()
        print(end - start)
        np.save('Quantile/empirical_quantile_normal/AW_thompson.npy', RESULT)

    # simulate emprical critical point
    for N in range(3, 41):
        print(N)
        start = time.perf_counter()
        for T in range(3, 41):
            myparams = {'N': N, 'T': T, 'R': sep_R, 'mean_reward': [0, 0], 'var_reward': [1, 1],
                        'clip': 0.1, 'algo': 'thompson', 'rwdtype': 'normal'}
            for i in range(ite):
                mbit = Bandit(params=myparams, cuda_available=True)
                est = mbit.regular_est().cpu()
                results[range(sep_R * i, (i + 1) * sep_R)] = est
            results.sort()
            RESULT[N, T, :9] = results[(total_R * quantile_point).astype(int)]
            RESULT[N, T, 9] = (RESULT[N, T, 7] - RESULT[N, T, 1]) / 2
        end = time.perf_counter()
        print(end - start)
        np.save('Quantile/empirical_quantile_normal/OLS_thompson.npy', RESULT)
