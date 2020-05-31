import numpy as np
import torch
import scipy.stats
import matplotlib.pyplot as plt
import time


class Bandit:

    def __init__(self, params=None, cuda_available=False):
        # N: batch size; T: batch number; R: replication number; clip: clip probability
        if params is None:
            params = {'N': 15, 'T': 25, 'R': 10000, 'mean_reward': [0, 0], 'var_reward': [1, 1],
                      'clip': 0.1, 'algo': 'thompson'}
        self.N = params['N']
        self.T = params['T']
        self.R = params['R']
        self.clip = params['clip']
        self.algo = params['algo']
        self.mean_rwd_env = torch.tensor(params['mean_reward'], dtype=torch.float)
        self.var_rwd_env = torch.tensor(params['var_reward'], dtype=torch.float)
        self.num_act_env = len(self.mean_rwd_env)
        if cuda_available:
            # temp variable for generating random variable
            self.rand_temp = torch.zeros((self.R, self.N), dtype=torch.float).cuda()
            self.crt_rwd = torch.zeros((self.R, self.N), dtype=torch.float).cuda()  # current reward
            self.crt_act = -torch.ones((self.R, self.N), dtype=torch.int).cuda()  # current action
            # current observation time for each action
            self.crt_obs= torch.zeros((self.R, self.num_act_env), dtype=torch.float).cuda()
            # accumulate mean reward estimation for each action
            self.acu_mean = torch.zeros((self.R, self.num_act_env), dtype=torch.float).cuda()
            # accumulate observation times for each action
            self.acu_obs = torch.zeros((self.R, self.num_act_env), dtype=torch.float).cuda()

        else:
            self.rand_temp = torch.zeros((self.R, self.N), dtype=torch.float)
            self.crt_rwd = torch.zeros((self.R, self.N), dtype=torch.float)
            self.crt_obs= torch.zeros((self.R, self.num_act_env), dtype=torch.float)
            # accumulate mean reward estimation for each action
            self.crt_act = -torch.ones((self.R, self.N), dtype=torch.int)
            self.acu_mean = torch.zeros((self.R,self.num_act_env), dtype=torch.float)
            self.acu_obs = torch.zeros((self.R, self.num_act_env), dtype=torch.float)

    def normal_rwd(self):
        # compute reward and update current mean estimation, obs time.
        for a in range(self.num_act_env):
            self.crt_rwd[self.crt_act == a] = self.rand_temp[self.crt_act == a].normal_(
                mean=self.mean_rwd_env[a], std=self.var_rwd_env[a].sqrt())
            self.crt_obs[:, a] = (self.crt_act == a).sum(dim=1)
            self.acu_mean[:, a] = ((self.crt_rwd * (self.crt_act == a).type(torch.float)).sum(
                dim=1) + self.acu_mean[:, a] * self.acu_obs[:, a]) / (
                    self.crt_obs[:, a]+self.acu_obs[:, a])
        self.acu_obs += self.crt_obs

    def first_step(self):
        self.rand_temp.uniform_(to=self.num_act_env)  # randomize action
        self.crt_act[:, :] = self.rand_temp.floor()
        # avoid non assignment; Only works for two arms.
        self.crt_act[(self.crt_act.sum(dim=1) == 0), 0] = 1
        self.crt_act[(self.crt_act.sum(dim=1) == self.N), 0] = 0
        self.normal_rwd()

    def step(self):
        if self.algo == 'thompson':
            p_one_better = self.thompson()
        elif self.algo == 'greedy':
            p_one_better = self.greedy()
        else:
            raise ValueError('algorithm not available')

        self.rand_temp.uniform_()  # randomize action
        self.crt_act[self.rand_temp < p_one_better.unsqueeze(dim=1)] = 1
        self.crt_act[self.rand_temp > p_one_better.unsqueeze(dim=1)] = 0
        self.crt_act[(self.crt_act.sum(dim=1) == 0), 0] = 1
        self.crt_act[(self.crt_act.sum(dim=1) == self.N), 0] = 0
        self.normal_rwd()

    def thompson(self):
        # only work for 2 arm bandit
        if self.acu_mean.is_cuda:
            temp = ((self.acu_mean[:, 1] - self.acu_mean[:, 0]) / ((1/self.acu_obs).sum(dim=1)).sqrt()).cpu()
            p_one_better = torch.Tensor(scipy.stats.norm.cdf(temp)).cuda()
            p_one_better[p_one_better < self.clip] = self.clip
            p_one_better[p_one_better > 1-self.clip] = 1 - self.clip
        else:
            temp = ((self.acu_mean[:, 1] - self.acu_mean[:, 0]) / ((1 / self.acu_obs).sum(dim=1)).sqrt())
            p_one_better = torch.Tensor(scipy.stats.norm.cdf(temp))
            p_one_better[p_one_better < self.clip] = self.clip
            p_one_better[p_one_better > 1-self.clip] = 1 - self.clip
        return p_one_better

    def greedy(self):
        if self.acu_mean.is_cuda:
            # Todo
            return
        else:
            p_one_better = torch.zeros(self.R)
            p_one_better[self.acu_mean.argmax(dim=1) == 1] = 1 - self.clip
            p_one_better[self.acu_mean.argmax(dim=1) == 0] = self.clip
        return p_one_better

    def regular_est(self):
        total_rwd = torch.zeros((self.R, self.N, self.T))
        total_act = torch.zeros((self.R, self.N, self.T))
        self.total_rwd = total_rwd
        self.total_act = total_act
        self.first_step()
        total_rwd[:, :, 0] = self.crt_rwd
        total_act[:, :, 0] = self.crt_act
        for t in range(1, self.T):
            self.step()
            total_rwd[:, :, t] = self.crt_rwd
            total_act[:, :, t] = self.crt_act
        if self.acu_mean.is_cuda:
            self.acu_mean = self.acu_mean.cpu()
            self.acu_obs = self.acu_obs.cpu()
        self.sigma_hat_square = ((total_rwd - total_act * self.acu_mean[:,1].reshape((self.R, 1, 1)) -
                                (1 - total_act) * self.acu_mean[:, 0].reshape((self.R, 1, 1))) ** 2
                                 ).sum(dim=(1, 2)) / (self.T * self.N - 2)
        self.diff = (self.acu_mean[:, 1] - self.acu_mean[:, 0])/self.sigma_hat_square.sqrt()/ (
                1 / self.acu_obs).sum(dim=(1)).sqrt()




# TODO cuda version


myparams = {'N': 100, 'T': 25, 'R': 50000, 'mean_reward': [0, 0], 'var_reward': [1, 1],
            'clip': 0, 'algo': 'thompson'}
mbit = Bandit(params=myparams,cuda_available=True)
mbit.regular_est()
mbit.first_step()
mbit.step(1)
mbit.thompson()


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
