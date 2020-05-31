import matplotlib.pyplot as plt
import time
import numpy as np
import scipy.stats
import Bandit_env.Bandit_env

start = time.perf_counter()
myparams = {'N': 100, 'T': 25, 'R': 100000, 'mean_reward': [0, 0], 'var_reward': [1, 1],
            'clip': 0, 'algo': 'thompson'}
mbit = Bandit(params=myparams,cuda_available=True)
mbit.regular_est()
end = time.perf_counter()
print(end-start)


plt.hist(mbit.diff,density=True, bins=80)
x_temp = np.arange(-4,4,0.02)
y_temp = scipy.stats.norm.pdf(x_temp)
plt.plot(x_temp,y_temp)
plt.show()
print("Type-1 error rate", (mbit.diff.abs()>scipy.stats.norm.ppf(q=1-0.05/2)).sum() /(0.0+mbit.R))
