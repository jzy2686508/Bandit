import torch
import scipy.stats
import numpy as np


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
            self.crt_obs = torch.zeros((self.R, self.num_act_env), dtype=torch.float).cuda()
            # accumulate mean reward estimation for each action
            self.acu_mean = torch.zeros((self.R, self.num_act_env), dtype=torch.float).cuda()
            # accumulate observation times for each action
            self.acu_obs = torch.zeros((self.R, self.num_act_env), dtype=torch.float).cuda()
            self.p_one_better = torch.ones((self.R), dtype=torch.float).cuda() / self.num_act_env
        else:
            self.rand_temp = torch.zeros((self.R, self.N), dtype=torch.float)
            self.crt_rwd = torch.zeros((self.R, self.N), dtype=torch.float)
            self.crt_obs = torch.zeros((self.R, self.num_act_env), dtype=torch.float)
            # accumulate mean reward estimation for each action
            self.crt_act = -torch.ones((self.R, self.N), dtype=torch.int)
            self.acu_mean = torch.zeros((self.R, self.num_act_env), dtype=torch.float)
            self.acu_obs = torch.zeros((self.R, self.num_act_env), dtype=torch.float)
            self.p_one_better = torch.ones((self.R), dtype=torch.float) / self.num_act_env

    def normal_rwd(self):
        # compute reward and update current mean estimation, obs time.
        for a in range(self.num_act_env):
            self.crt_rwd[self.crt_act == a] = self.rand_temp[self.crt_act == a].normal_(
                mean=self.mean_rwd_env[a], std=self.var_rwd_env[a].sqrt())
            self.crt_obs[:, a] = (self.crt_act == a).sum(dim=1)
            self.acu_mean[:, a] = ((self.crt_rwd * (self.crt_act == a).type(torch.float)).sum(
                dim=1) + self.acu_mean[:, a] * self.acu_obs[:, a]) / (
                                          self.crt_obs[:, a] + self.acu_obs[:, a])
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

        self.p_one_better = p_one_better
        self.rand_temp.uniform_()  # randomize action
        self.crt_act[self.rand_temp < p_one_better.unsqueeze(dim=1)] = 1
        self.crt_act[self.rand_temp > p_one_better.unsqueeze(dim=1)] = 0
        self.crt_act[(self.crt_act.sum(dim=1) == 0), 0] = 1
        self.crt_act[(self.crt_act.sum(dim=1) == self.N), 0] = 0
        self.normal_rwd()

    def thompson(self):
        # only work for 2 arm bandit
        if self.acu_mean.is_cuda:
            temp = ((self.acu_mean[:, 1] - self.acu_mean[:, 0]) / ((1 / self.acu_obs).sum(dim=1)).sqrt()).cpu()
            p_one_better = torch.Tensor(scipy.stats.norm.cdf(temp)).cuda()
            p_one_better[p_one_better < self.clip] = self.clip
            p_one_better[p_one_better > 1 - self.clip] = 1 - self.clip
        else:
            temp = ((self.acu_mean[:, 1] - self.acu_mean[:, 0]) / ((1 / self.acu_obs).sum(dim=1)).sqrt())
            p_one_better = torch.Tensor(scipy.stats.norm.cdf(temp))
            p_one_better[p_one_better < self.clip] = self.clip
            p_one_better[p_one_better > 1 - self.clip] = 1 - self.clip
        return p_one_better

    def greedy(self):
        if self.acu_mean.is_cuda:
            p_one_better = torch.zeros(self.R).cuda()
            p_one_better[self.acu_mean.argmax(dim=1) == 1] = 1 - self.clip
            p_one_better[self.acu_mean.argmax(dim=1) == 0] = self.clip
        else:
            p_one_better = torch.zeros(self.R)
            p_one_better[self.acu_mean.argmax(dim=1) == 1] = 1 - self.clip
            p_one_better[self.acu_mean.argmax(dim=1) == 0] = self.clip

        return p_one_better

    def regular_est(self):
        total_rwd = torch.zeros((self.R, self.N, self.T))
        total_act = torch.zeros((self.R, self.N, self.T))
        if self.acu_mean.is_cuda:
            total_act = total_act.cuda()
            total_rwd = total_rwd.cuda()
        self.first_step()
        total_rwd[:, :, 0] = self.crt_rwd
        total_act[:, :, 0] = self.crt_act
        for t in range(1, self.T):
            self.step()
            total_rwd[:, :, t] = self.crt_rwd
            total_act[:, :, t] = self.crt_act
        # if self.acu_mean.is_cuda:
        #     self.acu_mean = self.acu_mean.cpu()
        #     self.acu_obs = self.acu_obs.cpu()
        sigma_hat_square = ((total_rwd - total_act * self.acu_mean[:, 1].reshape((self.R, 1, 1)) -
                             (1 - total_act) * self.acu_mean[:, 0].reshape((self.R, 1, 1))) ** 2
                            ).sum(dim=(1, 2)) / (self.T * self.N - 2)
        # self.sigma_hat_square[:] = 1 #known variance
        return (self.acu_mean[:, 1] - self.acu_mean[:, 0]) / sigma_hat_square.sqrt() / (
                1 / self.acu_obs).sum(dim=1).sqrt()

    def batched_ols(self):
        bols = torch.zeros((self.R, self.T, self.num_act_env))
        bols_est = torch.zeros((self.R, self.T))
        if self.acu_mean.is_cuda:
            bols = bols.cuda()
            bols_est = bols_est.cuda()
        self.first_step()
        t = 0
        crt_act_float = self.crt_act.type(torch.float)
        bols[:, t, 1] = (self.crt_rwd * crt_act_float).sum(dim=1) / self.crt_obs[:, 1]
        bols[:, t, 0] = (self.crt_rwd * (1 - crt_act_float)).sum(dim=1) / self.crt_obs[:, 0]
        sigma_hat_square_bols = ((self.crt_rwd - crt_act_float * bols[:, t, 1].unsqueeze(dim=1) - (1 - crt_act_float) *
                                  bols[:, t, 0].unsqueeze(dim=1)) ** 2).sum(dim=1) / (self.N - 2)
        bols_est[:, t] = (bols[:, t, 1] - bols[:, t, 0]) * (self.crt_obs.prod(dim=1) / self.N /
                                                            sigma_hat_square_bols).sqrt()
        for t in range(1, self.T):
            self.step()
            crt_act_float = self.crt_act.type(torch.float)
            bols[:, t, 1] = (self.crt_rwd * crt_act_float).sum(dim=1) / self.crt_obs[:, 1]
            bols[:, t, 0] = (self.crt_rwd * (1 - crt_act_float)).sum(dim=1) / self.crt_obs[:, 0]
            sigma_hat_square_bols = ((self.crt_rwd - crt_act_float * bols[:, t, 1].unsqueeze(dim=1) -
                                      (1 - crt_act_float) * bols[:, t, 0].unsqueeze(dim=1)) ** 2).sum(dim=1) / (
                                            self.N - 2)
            bols_est[:, t] = (bols[:, t, 1] - bols[:, t, 0]) * (self.crt_obs.prod(dim=1) / self.N /
                                                                sigma_hat_square_bols).sqrt()
        test_stat = bols_est.mean(dim=1) * np.sqrt(self.T)
        self.bols = bols
        self.sigma = sigma_hat_square_bols
        return test_stat

    def aw_aipw(self):
        # TODO improve efficiency
        total_select_prob = torch.zeros((self.R, self.T, self.num_act_env))
        total_y1 = torch.zeros((self.R, self.N, self.T))
        total_y0 = torch.zeros((self.R, self.N, self.T))
        mu_hat = torch.zeros((self.R, self.num_act_env))
        if self.acu_mean.is_cuda:
            total_select_prob = total_select_prob.cuda()
            mu_hat = mu_hat.cuda()
            total_y1 = total_y1.cuda()
            total_y0 = total_y0.cuda()

        self.total_select_prob = total_select_prob
        self.mu_hat = mu_hat

        t = 0
        mu_hat[:, :] = self.acu_mean  # add mu hat
        self.first_step()
        crt_act_float = self.crt_act.type(torch.float)
        total_select_prob[:, t, 1] = self.p_one_better
        total_select_prob[:, t, 0] = 1 - self.p_one_better
        total_y1[:, :, t] = (self.crt_rwd * crt_act_float) / self.p_one_better.unsqueeze(dim=1) + \
                            (1 - crt_act_float / self.p_one_better.unsqueeze(dim=1)) * mu_hat[:, 1].unsqueeze(dim=1)
        total_y0[:, :, t] = (self.crt_rwd * (1 - crt_act_float)) / (1 - self.p_one_better).unsqueeze(dim=1) + \
                            (1 - (1 - crt_act_float) / (1 - self.p_one_better).unsqueeze(dim=1)) * mu_hat[:,
                                                                                                   0].unsqueeze(dim=1)
        for t in range(1, self.T):
            mu_hat[:, :] = self.acu_mean  # add mu hat
            self.step()
            crt_act_float = self.crt_act.type(torch.float)
            total_select_prob[:, t, 1] = self.p_one_better
            total_select_prob[:, t, 0] = 1 - self.p_one_better
            total_y1[:, :, t] = (self.crt_rwd * crt_act_float) / self.p_one_better.unsqueeze(dim=1) + \
                                (1 - crt_act_float / self.p_one_better.unsqueeze(dim=1)) * mu_hat[:, 1].unsqueeze(dim=1)
            total_y0[:, :, t] = (self.crt_rwd * (1 - crt_act_float)) / (1 - self.p_one_better).unsqueeze(dim=1) + \
                                (1 - (1 - crt_act_float) / (1 - self.p_one_better).unsqueeze(dim=1)) * mu_hat[:,
                                                                                                       0].unsqueeze(
                dim=1)
        beta_1_hat = (total_select_prob[:, :, 1].sqrt().unsqueeze(dim=1) * total_y1).sum(
            dim=(1, 2)) / total_select_prob[:, :, 1].sqrt().sum(dim=1) / self.N
        beta_0_hat = (total_select_prob[:, :, 0].sqrt().unsqueeze(dim=1) * total_y0).sum(
            dim=(1, 2)) / total_select_prob[:, :, 0].sqrt().sum(dim=1) / self.N
        v1_hat = (total_select_prob[:, :, 1].unsqueeze(dim=1) * ((total_y1 - beta_1_hat.reshape((self.R, 1, 1))) ** 2)).sum(dim=(1,2)) / \
                      (total_select_prob[:, :, 1].sqrt().sum(dim=1) * self.N) ** 2
        v0_hat = (total_select_prob[:, :, 0].unsqueeze(dim=1) * ((total_y0 - beta_0_hat.reshape((self.R, 1, 1))) ** 2)).sum(dim=(1,2)) / \
                      (total_select_prob[:, :, 0].sqrt().sum(dim=1) * self.N) ** 2
        cov01_hat = (total_select_prob.prod(dim=2).sqrt().unsqueeze(dim=1) * (total_y1 - beta_1_hat.reshape((self.R, 1, 1))) *
                     (total_y0 - beta_0_hat.reshape((self.R, 1, 1)))).sum(dim=(1,2)) / (total_select_prob[:, :, 1].sqrt().sum(dim=1) * self.N) / \
                    (total_select_prob[:, :, 0].sqrt().sum(dim=1) * self.N)
        # cov01_hat = (total_select_prob.prod(dim=2).sqrt() * (total_y0 - beta_0_hat.reshape((self.R, 1, 1))) * (
        #             total_y1 - beta_1_hat.reshape((self.R, 1, 1)))).sum(dim=(1,2)) / (
        #                              total_select_prob[:, :, 0].sqrt().sum(dim=1) * self.N) / (
        #                              total_select_prob[:, :, 1].sqrt().sum(dim=1) * self.N)
        self.cov01_hat = cov01_hat
        self.v0_hat = v0_hat
        self.v1_hat = v1_hat
        diff_est = (beta_1_hat - beta_0_hat) / (v0_hat + v1_hat - 2 * cov01_hat).sqrt()
        return diff_est
        # weight: sqrt(selection probability)
