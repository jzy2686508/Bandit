import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import scipy.stats
from Bandit_env.Bandit_env import Bandit



ecv_BOLS_greedy = np.load('../Quantile/empirical_quantile_normal/BOLS_greedy.npy')
ecv_BOLS_thompson = np.load('../Quantile/empirical_quantile_normal/BOLS_thompson.npy')
cv_t = np.load("../Quantile/quantile_standardized_t/quantile.npy")
critical_value = scipy.stats.norm.ppf(q=1-0.05/2)


torch.cuda.set_device(1)
total_R = 10000
ite = 1
sep_R = total_R // ite
results = np.zeros(total_R)

def type1(results):
    # power/ type1 error
    type1error = "Type-1 error / power:\n" + str((results.__abs__() > critical_value).sum() / results.size)
    print(type1error)
    type1error_adjust = "Adjust Type-1 error / power:\n" + str((results.__abs__() > cv_t[N, T, 9]).sum() / results.size)
    print(type1error_adjust)
    if algo == 'greedy':
        type1error_ecv = "Type-1 error / power with empirical cutoff:\n" + str(
            (results.__abs__() > ecv_BOLS_greedy[N, T, 9]).sum() / results.size)
    elif algo == 'thompson':
        type1error_ecv = "Type-1 error / power with empirical cutoff:\n" + str(
            (results.__abs__() > ecv_BOLS_thompson[N, T, 9]).sum() / results.size)
    print(type1error_ecv)
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
    # plt.show()
    return f


N = 25
T = 15
rwdtype = 'normal'
est_method = 'BOLS'  # 'AW', 'BOLS', 'OLS'
algo = 'thompson'
myparams = {'N': N, 'T': T, 'R': sep_R, 'mean_reward': [0, 0.5], 'var_reward': [1, 1],
            'clip': 0.05, 'algo': algo, 'rwdtype': rwdtype}
start = time.perf_counter()
for i in range(ite):
    mbit = Bandit(params=myparams, cuda_available=True)
    mbit.cpt_exp(0,0.25)
    if est_method=='AW':
        est = mbit.aw_aipw().cpu()
    elif est_method=='BOLS':
        est = mbit.batched_ols().cpu()
    elif est_method=='OLS':
        est = mbit.regular_est().cpu()
    results[range(sep_R * i, (i+1) * sep_R)] = est
    results2 = mbit.result_weight.cpu().numpy()
end = time.perf_counter()
print(end-start)

f1 = type1(results)
# f1.show()
print("***************************************************************")
f2 = type1(results2)
# f2.show()
#
# for N in [25]:
#     for T in [5,10,15,20,25]:
#         algo = 'greedy'
#         myparams = {'N': N, 'T': T, 'R': sep_R, 'mean_reward': [0.25, 0], 'var_reward': [1, 1],
#                     'clip': 0.05, 'algo': algo, 'rwdtype': rwdtype}
#         start = time.perf_counter()
#         for i in range(ite):
#             mbit = Bandit(params=myparams, cuda_available=True)
#             if est_method == 'AW':
#                 est = mbit.aw_aipw().cpu()
#             elif est_method == 'BOLS':
#                 est = mbit.batched_ols().cpu()
#             elif est_method == 'OLS':
#                 est = mbit.regular_est().cpu()
#             results[range(sep_R * i, (i + 1) * sep_R)] = est
#             results2 = mbit.result_weight.cpu().numpy()
#         end = time.perf_counter()
#         print(N,T)
#         type1(results)
#         type1(results2)
