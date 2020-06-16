import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import scipy.stats
from Bandit_env.Bandit_env import Bandit



critical_value = scipy.stats.norm.ppf(q=1-0.05/2)

torch.cuda.set_device(3)
total_R = 200000
ite = 1
sep_R = total_R // ite
results = np.zeros(total_R)


def type1(results,N,T):
    # power/ type1 error

    type1error = "Type-1 error / power:\n" + str((results.__abs__() > critical_value).sum() / results.size)
    print(type1error)
    # if algo == 'greedy':
    #     type1error_ecv = "Type-1 error / power with empirical cutoff:\n" + str(
    #         (results.__abs__() > ecv_AW_greedy[N, T, 9]).sum() / results.size)
    # elif algo == 'thompson':
    #     type1error_ecv = "Type-1 error / power with empirical cutoff:\n" + str(
    #         (results.__abs__() > ecv_AW_thompson[N, T, 9]).sum() / results.size)
    # print(type1error_ecv)

    # plot distribution
    f = plt.figure()
    str_label = est_method + '  N=' + str(N) + ' T=' + str(T) + \
                ' ' + str(myparams['algo']) + ' clip=' + str(myparams['clip'])
    plt.hist(results, density=True, bins=150, label=type1error)
    x_temp = np.arange(-4, 4, 0.02)
    y_temp = scipy.stats.norm.pdf(x_temp)
    plt.plot(x_temp, y_temp, label='Standard Normal')
    plt.title(str_label)
    plt.legend()
    return f

N = 15
T = 15
rwdtype = 'normal'
algo = 'thompson'
est_method = 'AW-DIFF'
myparams = {'N': N, 'T': T, 'R': sep_R, 'mean_reward': [10, 10.25], 'var_reward': [1, 1],
            'clip': 0.1, 'algo': algo, 'rwdtype': rwdtype}
start = time.perf_counter()
for i in range(ite):
    mbit = Bandit(params=myparams, cuda_available=True)
    est = mbit.aw_diff().cpu().numpy()
    results[range(sep_R * i, (i+1) * sep_R)] = est
end = time.perf_counter()
print(end-start)

f1 = type1(results,myparams['N'],myparams['T'])
f1.show()
# results_0 = ((mbit.beta0 - myparams['mean_reward'][0]) / mbit.v0_hat.sqrt()).cpu().numpy()
# f2 = type1(results_0,myparams['N'],myparams['T'])
# f2.show()
# temp = (mbit.v1_hat * (mbit.total_prob[:,:,1].sqrt().sum(dim=1))**2) / mbit.T / (mbit.var_rwd_env[1] + 0.5 * 10**2)
# temp2 = (mbit.v1_hat * (mbit.total_prob[:,:,1].sqrt().sum(dim=1))**2) / (mbit.T * (1.333 + 100) - 100*mbit.total_prob[:,:,1].sum(dim=1))


# results_1 = (mbit.beta0 / mbit.beta0.std()).cpu().numpy()
# f3 = type1(results_1)
# f3.show()

# plt.hist(mbit.mu_hat[:,0].cpu(), bins=200)
# plt.show()
# plt.hist(mbit.beta0[:].cpu(), bins=200, density=True)
# plt.show()
# print("*********************************************************")
# results2 = results / np.sqrt((N*T)/(N*T-1))
# f2 = type1(results2)


# temp = np.zeros((11,11, total_R))
# temp1 = []
# for ite1 in range(1, 10):
#     for ite2 in range(1, 10):
#         N = ite1 * 5
#         T = ite2 * 5
#         print(ite1,ite2)
#         rwdtype = 'normal'
#         est_method = 'AW'  # 'AW', 'BOLS', 'OLS'
#         algo = 'thompson'
#         myparams = {'N': N, 'T': T, 'R': sep_R, 'mean_reward': [0, 0], 'var_reward': [1, 1],
#                     'clip': 0.1, 'algo': algo, 'rwdtype': rwdtype}
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
#         end = time.perf_counter()
#         print(end - start)
#         temp[ite1, ite2, :] = (mbit.beta0 / mbit.v0_hat.sqrt()).cpu().numpy()
#         temp1.append(type1(temp[ite1,ite2,:], myparams['N'],myparams['T']))


# temp1[53].savefig("../Figure/temp/0608_8.pdf")