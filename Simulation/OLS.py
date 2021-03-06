import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import scipy.stats
from Bandit_env.Bandit_env import Bandit


ecv_OLS_greedy = np.load('../Quantile/empirical_quantile_normal/OLS_greedy.npy')
ecv_OLS_thompson = np.load('../Quantile/empirical_quantile_normal/OLS_thompson.npy')
critical_value = scipy.stats.norm.ppf(q=1-0.05/2)


torch.cuda.set_device(3)
total_R = 100000
ite = 1
sep_R = total_R // ite
results = np.zeros(total_R)

def type1(results):
    # power/ type1 error

    type1error = "Type-1 error / power:\n" + str((results.__abs__() > critical_value).sum() / results.size)
    print(type1error)
    # if algo == 'greedy':
    #     type1error_ecv = "Type-1 error / power with empirical cutoff:\n" + str(
    #         (results.__abs__() > ecv_OLS_greedy[N, T, 9]).sum() / results.size)
    # elif algo == 'thompson':
    #     type1error_ecv = "Type-1 error / power with empirical cutoff:\n" + str(
    #         (results.__abs__() > ecv_OLS_thompson[N, T, 9]).sum() / results.size)
    # print(type1error_ecv)

    # plot distribution
    f = plt.figure()
    str_label = est_method + '  N=' + str(myparams['N']) + ' T=' + str(myparams['T']) + \
                ' ' + str(myparams['algo']) + ' clip=' + str(myparams['clip'])
    plt.hist(results, density=True, bins=200, label=type1error)
    x_temp = np.arange(-4, 4, 0.02)
    y_temp = scipy.stats.norm.pdf(x_temp)
    plt.plot(x_temp, y_temp, label='Standard Normal')
    plt.title(str_label)
    plt.legend()
    return f

N = 100
T = 25
rwdtype = 'normal'
est_method = 'OLS'  # 'AW', 'BOLS', 'OLS'
algo = 'greedy'
myparams = {'N': N, 'T': T, 'R': sep_R, 'mean_reward': [0, 0], 'var_reward': [1, 1],
            'clip': 0.1, 'algo': algo, 'rwdtype': rwdtype}
start = time.perf_counter()
for i in range(ite):
    mbit = Bandit(params=myparams, cuda_available=True)
    if est_method=='AW':
        est = mbit.aw_aipw().cpu()
    elif est_method=='BOLS':
        est = mbit.batched_ols().cpu()
    elif est_method=='OLS':
        est = mbit.regular_est().cpu()
    results[range(sep_R * i, (i+1) * sep_R)] = est
end = time.perf_counter()
print(end-start)

f1 = type1(results)
f1.show()