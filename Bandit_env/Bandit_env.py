import torch
import scipy.stats
import numpy as np


class Bandit:

    def __init__(self, params=None, cuda_available=False):
        # N: batch size; T: batch number; R: replication number; clip: clip probability
        if params is None:
            params = {'N': 15, 'T': 25, 'R': 10000, 'mean_reward': [0, 0], 'var_reward': [1, 1],
                      'clip': 0.1, 'algo': 'thompson', 'rwdtype': 'normal'}
        self.normalcdf = torch.distributions.normal.Normal(0,1)
        self.N = params['N']
        self.T = params['T']
        self.R = params['R']
        self.clip = params['clip']
        self.algo = params['algo']
        self.mean_rwd_env = torch.tensor(params['mean_reward'], dtype=torch.float)
        self.var_rwd_env = torch.tensor(params['var_reward'], dtype=torch.float)
        self.num_act_env = len(self.mean_rwd_env)
        self.is_uniform = (params['rwdtype']=='uniform')
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
            # total reward
            # self.total_rwd = torch.zeros((self.R, self.N, self.T), dtype=torch.float).cuda()
            # total selection probability
            self.total_prob = torch.zeros((self.R, self.T, self.num_act_env), dtype=torch.float).cuda()
            # total action
            # self.total_act = torch.zeros((self.R, self.N, self.T), dtype=torch.float).cuda()
            self.exp_select_prob = torch.zeros((self.R, self.T), dtype=torch.float).cuda()
        else:
            self.rand_temp = torch.zeros((self.R, self.N), dtype=torch.float)
            self.crt_rwd = torch.zeros((self.R, self.N), dtype=torch.float)
            self.crt_obs = torch.zeros((self.R, self.num_act_env), dtype=torch.float)
            # accumulate mean reward estimation for each action
            self.crt_act = -torch.ones((self.R, self.N), dtype=torch.int)
            self.acu_mean = torch.zeros((self.R, self.num_act_env), dtype=torch.float)
            self.acu_obs = torch.zeros((self.R, self.num_act_env), dtype=torch.float)
            self.p_one_better = torch.ones((self.R), dtype=torch.float) / self.num_act_env
            # self.total_rwd = torch.zeros((self.R, self.N, self.T), dtype=torch.float)
            self.total_prob = torch.zeros((self.R, self.T, self.num_act_env), dtype=torch.float)
            # self.total_act = torch.zeros((self.R, self.N, self.T), dtype=torch.float)
            self.exp_select_prob = torch.zeros((self.R, self.T), dtype=torch.float)

    def get_rwd(self):
        # compute reward and update current mean estimation, obs time.
        # Uniform [0,1]
        if self.is_uniform:
            for a in range(self.num_act_env):
                self.crt_rwd[self.crt_act == a] = self.rand_temp[self.crt_act == a].uniform_(self.mean_rwd_env[a] -
                    (self.var_rwd_env[a] * 12).sqrt()/2, self.mean_rwd_env[a] + (self.var_rwd_env[a] * 12).sqrt()/2)
                self.crt_obs[:, a] = (self.crt_act == a).sum(dim=1)
                self.acu_mean[:, a] = ((self.crt_rwd * (self.crt_act == a).type(torch.float)).sum(
                    dim=1) + self.acu_mean[:, a] * self.acu_obs[:, a]) / (
                                              self.crt_obs[:, a] + self.acu_obs[:, a] + 1e-10)
            self.acu_obs += self.crt_obs
        else:
            for a in range(self.num_act_env):
                self.crt_rwd[self.crt_act == a] = self.rand_temp[self.crt_act == a].normal_(
                    mean=self.mean_rwd_env[a], std=self.var_rwd_env[a].sqrt())
                self.crt_obs[:, a] = (self.crt_act == a).sum(dim=1)
                self.acu_mean[:, a] = ((self.crt_rwd * (self.crt_act == a).type(torch.float)).sum(
                    dim=1) + self.acu_mean[:, a] * self.acu_obs[:, a]) / (
                                              self.crt_obs[:, a] + self.acu_obs[:, a] + 1e-10)
            self.acu_obs += self.crt_obs

    def first_step(self, is_aw=False):
        self.rand_temp.uniform_(to=self.num_act_env)  # randomize action
        self.crt_act[:, :] = self.rand_temp.floor()
        # avoid non assignment; Only works for two arms.
        if not is_aw:
            self.crt_act[(self.crt_act.sum(dim=1) == 0), 0] = 1
            self.crt_act[(self.crt_act.sum(dim=1) == self.N), 0] = 0
        self.normal_rwd(uniform=self.is_uniform)

    def step(self, assign_1=False):
        self.rand_temp.uniform_()  # randomize action
        self.crt_act[self.rand_temp < self.p_one_better.unsqueeze(dim=1)] = 1
        self.crt_act[self.rand_temp > self.p_one_better.unsqueeze(dim=1)] = 0
        if assign_1:
            self.crt_act[(self.crt_act.sum(dim=1) == 0), 0] = 1
            self.crt_act[(self.crt_act.sum(dim=1) == self.N), 0] = 0

    def thompson(self):
        # only work for 2 arm bandit
        if self.acu_mean.is_cuda:
            temp = ((self.acu_mean[:, 1] - self.acu_mean[:, 0]) / ((1 / self.acu_obs).sum(dim=1)).sqrt()).cpu()
            self.p_one_better[:] = self.normalcdf.cdf(temp)
            self.p_one_better[self.p_one_better < self.clip] = self.clip
            self.p_one_better[self.p_one_better > 1 - self.clip] = 1 - self.clip
        else:
            temp = ((self.acu_mean[:, 1] - self.acu_mean[:, 0]) / ((1 / self.acu_obs).sum(dim=1)).sqrt())
            self.p_one_bette[:] = scipy.stats.norm.cdf(temp)
            self.p_one_better[self.p_one_better < self.clip] = self.clip
            self.p_one_better[self.p_one_better > 1 - self.clip] = 1 - self.clip
        return self.p_one_better

    def greedy(self):
        if self.acu_mean.is_cuda:
            self.p_one_better[self.acu_mean.argmax(dim=1) == 1] = 1 - self.clip
            self.p_one_better[self.acu_mean.argmax(dim=1) == 0] = self.clip
        else:
            self.p_one_better[self.acu_mean.argmax(dim=1) == 1] = 1 - self.clip
            self.p_one_better[self.acu_mean.argmax(dim=1) == 0] = self.clip

        return self.p_one_better

    def regular_est(self):
        # total reward & action
        if self.acu_mean.is_cuda:
            self.total_rwd = torch.zeros((self.R, self.N, self.T), dtype=torch.float).cuda()
            self.total_act = torch.zeros((self.R, self.N, self.T), dtype=torch.float).cuda()
        else:
            self.total_rwd = torch.zeros((self.R, self.N, self.T), dtype=torch.float)
            self.total_act = torch.zeros((self.R, self.N, self.T), dtype=torch.float)
        # first step
        self.step(assign_1=False)
        self.get_rwd()
        self.total_rwd[:, :, 0] = self.crt_rwd
        self.total_act[:, :, 0] = self.crt_act
        for t in range(1, self.T):
            if self.algo == 'thompson':
                self.thompson()
            elif self.algo == 'greedy':
                self.greedy()
            else:
                raise ValueError('algorithm not available')
            self.step(assign_1=False)
            self.get_rwd()
            self.total_rwd[:, :, t] = self.crt_rwd
            self.total_act[:, :, t] = self.crt_act
        sigma_hat_square = ((self.total_rwd - self.total_act * self.acu_mean[:, 1].reshape((self.R, 1, 1)) -
                             (1 - self.total_act) * self.acu_mean[:, 0].reshape((self.R, 1, 1))) ** 2
                            ).sum(dim=(1, 2)) / (self.T * self.N - 2)
        # sigma_hat_square[:] = 1 #known variance
        return (self.acu_mean[:, 1] - self.acu_mean[:, 0]) / sigma_hat_square.sqrt() / (
                1 / self.acu_obs).sum(dim=1).sqrt()

    def batched_ols(self):
        bols = torch.zeros((self.R, self.T, self.num_act_env))
        bols_est = torch.zeros((self.R, self.T))
        self.H = torch.ones((self.R, self.T))
        if self.acu_mean.is_cuda:
            bols = bols.cuda()
            bols_est = bols_est.cuda()
            self.H = self.H.cuda()
        H = self.H
        t = 0
        # thompson
        H[:, 0] = self.cpt_exp(t=t, beta=0.5)
        # # greedy
        # H[:, t] = (self.p_one_better * (1 - self.p_one_better)).sqrt() / (
        #         (self.p_one_better * (1 - self.p_one_better)).sqrt() + (self.T-1) * np.sqrt(self.clip * (1 - self.clip)))
        self.step(assign_1=True)
        self.get_rwd()
        crt_act_float = self.crt_act.type(torch.float)
        bols[:, t, 1] = (self.crt_rwd * crt_act_float).sum(dim=1) / self.crt_obs[:, 1]
        bols[:, t, 0] = (self.crt_rwd * (1 - crt_act_float)).sum(dim=1) / self.crt_obs[:, 0]
        sigma_hat_square_bols = ((self.crt_rwd - crt_act_float * bols[:, t, 1].unsqueeze(dim=1) - (1 - crt_act_float) *
                                  bols[:, t, 0].unsqueeze(dim=1)) ** 2).sum(dim=1) / (self.N - 2)
        bols_est[:, t] = (bols[:, t, 1] - bols[:, t, 0]) * (self.crt_obs.prod(dim=1) / self.N /
                                                            sigma_hat_square_bols).sqrt()
        self.total_prob[:, t, 1] = self.p_one_better
        self.total_prob[:, t, 0] = 1 - self.p_one_better

        for t in range(1, self.T):
            # thompson
            H[:, t] = (1 - H[:, :t].sum(dim=1))*self.cpt_exp(t=t,beta=0.5)
            # # greedy
            # H[:, t] = np.sqrt(self.clip * (1 - self.clip)) / (
            #   (self.p_one_better * (1 - self.p_one_better)).sqrt() + (self.T-1) * np.sqrt(self.clip * (1 - self.clip)))
            if self.algo == 'thompson':
                self.thompson()
            elif self.algo == 'greedy':
                self.greedy()
            else:
                raise ValueError('algorithm not available')
            self.step(assign_1=True)
            self.get_rwd()
            crt_act_float = self.crt_act.type(torch.float)
            bols[:, t, 1] = (self.crt_rwd * crt_act_float).sum(dim=1) / self.crt_obs[:, 1]
            bols[:, t, 0] = (self.crt_rwd * (1 - crt_act_float)).sum(dim=1) / self.crt_obs[:, 0]
            sigma_hat_square_bols = ((self.crt_rwd - crt_act_float * bols[:, t, 1].unsqueeze(dim=1) -
                                      (1 - crt_act_float) * bols[:, t, 0].unsqueeze(dim=1)) ** 2).sum(dim=1) / (
                                            self.N - 2)
            bols_est[:, t] = (bols[:, t, 1] - bols[:, t, 0]) * (self.crt_obs.prod(dim=1) / self.N /
                                                                sigma_hat_square_bols).sqrt()
            self.total_prob[:, t, 1] = self.p_one_better
            self.total_prob[:, t, 0] = 1 - self.p_one_better

        test_stat = bols_est.mean(dim=1) * np.sqrt(self.T)
        self.bols_est = bols_est
        self.sigma = sigma_hat_square_bols
        self.result_weight = (bols_est * H).sum(dim=1) / H.norm(dim=1)

        return test_stat

    def batched_IPW(self):
        bols = torch.zeros((self.R, self.T, self.num_act_env))
        bols_est = torch.zeros((self.R, self.T))
        self.H = torch.ones((self.R, self.T))
        if self.acu_mean.is_cuda:
            bols = bols.cuda()
            bols_est = bols_est.cuda()
            self.H = self.H.cuda()
        H = self.H
        t = 0
        # thompson
        H[:, 0] = self.cpt_exp(t=t, beta=0.5)
        # # greedy
        # H[:, t] = (self.p_one_better * (1 - self.p_one_better)).sqrt() / (
        #         (self.p_one_better * (1 - self.p_one_better)).sqrt() + (self.T-1) * np.sqrt(self.clip * (1 - self.clip)))
        self.step(assign_1=True)
        self.get_rwd()
        crt_act_float = self.crt_act.type(torch.float)
        bols[:, t, 1] = (self.crt_rwd * crt_act_float).sum(dim=1) / self.crt_obs[:, 1]
        bols[:, t, 0] = (self.crt_rwd * (1 - crt_act_float)).sum(dim=1) / self.crt_obs[:, 0]
        sigma_hat_square_bols = ((self.crt_rwd - crt_act_float * bols[:, t, 1].unsqueeze(dim=1) - (1 - crt_act_float) *
                                  bols[:, t, 0].unsqueeze(dim=1)) ** 2).sum(dim=1) / (self.N - 2)
        bols_est[:, t] = (bols[:, t, 1] - bols[:, t, 0]) * (self.crt_obs.prod(dim=1) / self.N /
                                                            sigma_hat_square_bols).sqrt()
        self.total_prob[:, t, 1] = self.p_one_better
        self.total_prob[:, t, 0] = 1 - self.p_one_better

        for t in range(1, self.T):
            # thompson
            H[:, t] = (1 - H[:, :t].sum(dim=1))*self.cpt_exp(t=t,beta=0.5)
            # # greedy
            # H[:, t] = np.sqrt(self.clip * (1 - self.clip)) / (
            #   (self.p_one_better * (1 - self.p_one_better)).sqrt() + (self.T-1) * np.sqrt(self.clip * (1 - self.clip)))
            if self.algo == 'thompson':
                self.thompson()
            elif self.algo == 'greedy':
                self.greedy()
            else:
                raise ValueError('algorithm not available')
            self.step(assign_1=True)
            self.get_rwd()
            crt_act_float = self.crt_act.type(torch.float)
            bols[:, t, 1] = (self.crt_rwd * crt_act_float).sum(dim=1) / self.crt_obs[:, 1]
            bols[:, t, 0] = (self.crt_rwd * (1 - crt_act_float)).sum(dim=1) / self.crt_obs[:, 0]
            sigma_hat_square_bols = ((self.crt_rwd - crt_act_float * bols[:, t, 1].unsqueeze(dim=1) -
                                      (1 - crt_act_float) * bols[:, t, 0].unsqueeze(dim=1)) ** 2).sum(dim=1) / (
                                            self.N - 2)
            bols_est[:, t] = (bols[:, t, 1] - bols[:, t, 0]) * (self.crt_obs.prod(dim=1) / self.N /
                                                                sigma_hat_square_bols).sqrt()
            self.total_prob[:, t, 1] = self.p_one_better
            self.total_prob[:, t, 0] = 1 - self.p_one_better

        test_stat = bols_est.mean(dim=1) * np.sqrt(self.T)
        self.bols_est = bols_est
        self.sigma = sigma_hat_square_bols
        self.result_weight = (bols_est * H).sum(dim=1) / H.norm(dim=1)

        return test_stat

    def aw_aipw(self):
        total_y1 = torch.zeros((self.R, self.N, self.T))
        total_y0 = torch.zeros((self.R, self.N, self.T))
        mu_hat = torch.zeros((self.R, self.num_act_env))
        Wt = torch.ones((self.R,self.T,self.num_act_env))
        if self.acu_mean.is_cuda:
            mu_hat = mu_hat.cuda()
            total_y1 = total_y1.cuda()
            total_y0 = total_y0.cuda()
            Wt = Wt.cuda()

        self.mu_hat = mu_hat

        t = 0
        mu_hat[:, :] = self.acu_mean  # add mu hat
        # mu_hat[:, :] = 0
        self.step(assign_1=False)
        self.get_rwd()
        crt_act_float = self.crt_act.type(torch.float)
        self.total_prob[:, t, 1] = self.p_one_better
        self.total_prob[:, t, 0] = 1 - self.p_one_better
        total_y1[:, :, t] = (self.crt_rwd * crt_act_float) / self.p_one_better.unsqueeze(dim=1) + \
                            (1 - crt_act_float / self.p_one_better.unsqueeze(dim=1)) * mu_hat[:, 1].unsqueeze(dim=1)
        total_y0[:, :, t] = (self.crt_rwd * (1 - crt_act_float)) / (1 - self.p_one_better).unsqueeze(dim=1) + \
                            (1 - (1 - crt_act_float) / (1 - self.p_one_better).unsqueeze(dim=1)) * mu_hat[:,
                                                                                                   0].unsqueeze(dim=1)
        for t in range(1, self.T):
            mu_hat[:, :] = self.acu_mean  # add mu hat
            # mu_hat[:, :] = 0
            if self.algo == 'thompson':
                self.thompson()
            elif self.algo == 'greedy':
                self.greedy()
            else:
                raise ValueError('algorithm not available')
            self.step(assign_1=False)
            self.get_rwd()
            crt_act_float = self.crt_act.type(torch.float)
            self.total_prob[:, t, 1] = self.p_one_better
            self.total_prob[:, t, 0] = 1 - self.p_one_better
            total_y1[:, :, t] = (self.crt_rwd * crt_act_float) / self.p_one_better.unsqueeze(dim=1) + \
                                (1 - crt_act_float / self.p_one_better.unsqueeze(dim=1)) * mu_hat[:, 1].unsqueeze(dim=1)
            total_y0[:, :, t] = (self.crt_rwd * (1 - crt_act_float)) / (1 - self.p_one_better).unsqueeze(dim=1) + \
                                (1 - (1 - crt_act_float) / (1 - self.p_one_better).unsqueeze(dim=1)) * mu_hat[:,
                                                                                                       0].unsqueeze(
                dim=1)
        Wt[:, :, 1] = self.total_prob[:, :, 1].sqrt()
        Wt[:, :, 0] = self.total_prob[:, :, 0].sqrt()
        # Wt[:, :, 1] = total_select_prob[:, :, 1]
        # Wt[:, :, 0] = total_select_prob[:, :, 0]
        # Wt[:, :, 1] = 1
        # Wt[:, :, 0] = 1
        beta_1_hat = (Wt[:, :, 1].unsqueeze(dim=1) * total_y1).sum(
            dim=(1, 2)) / Wt[:,:,1].sum(dim=1) / self.N
        beta_0_hat = (Wt[:, :, 0].unsqueeze(dim=1) * total_y0).sum(
            dim=(1, 2)) / Wt[:, :, 0].sum(dim=1) / self.N
        v1_hat = ((Wt[:, :, 1]**2).unsqueeze(dim=1) * ((total_y1 - beta_1_hat.reshape((self.R, 1, 1))) ** 2)).sum(dim=(1,2)) / \
                      (Wt[:, :, 1].sum(dim=1) * self.N) ** 2
        v0_hat = ((Wt[:, :, 0]**2).unsqueeze(dim=1) * ((total_y0 - beta_0_hat.reshape((self.R, 1, 1))) ** 2)).sum(dim=(1,2)) / \
                      (Wt[:, :, 0].sum(dim=1) * self.N) ** 2
        cov01_hat = (Wt.prod(dim=2).unsqueeze(dim=1) * (total_y1 - beta_1_hat.reshape((self.R, 1, 1))) *
                     (total_y0 - beta_0_hat.reshape((self.R, 1, 1)))).sum(dim=(1,2)) / (Wt[:, :, 1].sum(dim=1) * self.N) / \
                    (Wt[:, :, 0].sum(dim=1) * self.N)
        self.cov01_hat = cov01_hat
        self.v0_hat = v0_hat
        self.v1_hat = v1_hat
        self.beta1 = beta_1_hat
        self.beta0 = beta_0_hat
        self.y1 = total_y1
        diff_est = (beta_1_hat - beta_0_hat) / (v0_hat + v1_hat - 2 * cov01_hat).sqrt()
        return diff_est
        # weight: sqrt(selection probability)

    def aw_diff(self):
        total_y1 = torch.zeros((self.R, self.N, self.T))
        total_y0 = torch.zeros((self.R, self.N, self.T))
        total_gamma = torch.zeros((self.R, self.N, self.T))
        mu_hat = torch.zeros((self.R, self.num_act_env))
        Wt = torch.ones((self.R, self.T))
        if self.acu_mean.is_cuda:
            mu_hat = mu_hat.cuda()
            total_y1 = total_y1.cuda()
            total_y0 = total_y0.cuda()
            total_gamma = total_gamma.cuda()
            Wt = Wt.cuda()

        self.mu_hat = mu_hat
        self.Wt = Wt
        t = 0
        mu_hat[:, :] = self.acu_mean  # add mu hat
        # mu_hat[:, :] = 0
        self.step(assign_1=False)
        self.get_rwd()
        crt_act_float = self.crt_act.type(torch.float)
        self.total_prob[:, t, 1] = self.p_one_better
        self.total_prob[:, t, 0] = 1 - self.p_one_better
        total_y1[:, :, t] = (self.crt_rwd * crt_act_float) / self.p_one_better.unsqueeze(dim=1) + \
                            (1 - crt_act_float / self.p_one_better.unsqueeze(dim=1)) * mu_hat[:, 1].unsqueeze(dim=1)
        total_y0[:, :, t] = (self.crt_rwd * (1 - crt_act_float)) / (1 - self.p_one_better).unsqueeze(dim=1) + \
                            (1 - (1 - crt_act_float) / (1 - self.p_one_better).unsqueeze(dim=1)) * mu_hat[:,
                            0].unsqueeze(dim=1)
        total_gamma[:, :, t] = total_y1[:, :, t] - total_y0[:, :, t]
        for t in range(1, self.T):
            mu_hat[:, :] = self.acu_mean  # add mu hat
            # mu_hat[:, :] = 0
            if self.algo == 'thompson':
                self.thompson()
            elif self.algo == 'greedy':
                self.greedy()
            else:
                raise ValueError('algorithm not available')
            self.step(assign_1=False)
            self.get_rwd()
            crt_act_float = self.crt_act.type(torch.float)
            self.total_prob[:, t, 1] = self.p_one_better
            self.total_prob[:, t, 0] = 1 - self.p_one_better
            total_y1[:, :, t] = (self.crt_rwd * crt_act_float) / self.p_one_better.unsqueeze(dim=1) + \
                                (1 - crt_act_float / self.p_one_better.unsqueeze(dim=1)) * mu_hat[:, 1].unsqueeze(dim=1)
            total_y0[:, :, t] = (self.crt_rwd * (1 - crt_act_float)) / (1 - self.p_one_better).unsqueeze(dim=1) + \
                                (1 - (1 - crt_act_float) / (1 - self.p_one_better).unsqueeze(dim=1)) * mu_hat[:,
                                0].unsqueeze(dim=1)
            total_gamma[:, :, t] = total_y1[:, :, t] - total_y0[:, :, t]
        Wt[:,:] = self.total_prob.prod(dim=2).sqrt()
        diff_hat = (Wt.unsqueeze(dim=1) * total_gamma).sum(dim=(1,2)) / Wt.sum(dim=1) / self.N
        v_hat = ((Wt ** 2).unsqueeze(dim=1) * (total_gamma - diff_hat.reshape((self.R, 1, 1)))**2).sum(dim=(1,2)) / (Wt.sum(dim=1)**2*self.N**2)
        self.v_hat = v_hat
        self.diff_hat = diff_hat
        self.total_gamma = total_gamma
        diff_test = diff_hat / v_hat.sqrt()
        return diff_test

    def cpt_exp(self, t, beta):
        # set current prob
        # set current thompson prior
        crt_prior = torch.zeros((self.R, self.num_act_env))
        if self.acu_mean.is_cuda:
            crt_prior = crt_prior.cuda()
        self.crt_prior = crt_prior
        if t==0:
            crt_prior[:, 0] = 0
            crt_prior[:, 1] = 1e10
        else:
            crt_prior[:, 0] = self.acu_mean[:, 1] - self.acu_mean[:, 0]
            crt_prior[:, 1] = (1 / self.acu_obs).sum(dim=1)

        temp7 = self.normalcdf.cdf((crt_prior[:, 0] / crt_prior[:, 1].sqrt()))
        temp7[temp7 > 1 - self.clip] = 1 - self.clip
        temp7[temp7 < self.clip] = self.clip
        self.exp_select_prob[:, t] = temp7

        for i in range(t+1,self.T):
            # compute new thompson belief
            var_exp = 1 / self.exp_select_prob[:, i - 1] / (1 - self.exp_select_prob[:, i - 1]) / self.N
            crt_prior[:, 0] = crt_prior[:, 0] * var_exp / (crt_prior[:, 1] + var_exp) + \
                              beta * crt_prior[:, 1] / (crt_prior[:, 1] + var_exp)
            crt_prior[:, 1] = crt_prior[:, 1] * var_exp / (crt_prior[:, 1] + var_exp)
            temp7 = self.normalcdf.cdf((crt_prior[:, 0] / crt_prior[:, 1].sqrt()))
            temp7[temp7 > 1 - self.clip] = 1 - self.clip
            temp7[temp7 < self.clip] = self.clip
            self.exp_select_prob[:, i] = temp7

        # predict later selection probability
        return (self.exp_select_prob[:, t] * (1 - self.exp_select_prob[:,t])).sqrt() / (self.exp_select_prob[:, t:self.T] * (1 - self.exp_select_prob[:, t:self.T])).sqrt().sum(dim=1)


