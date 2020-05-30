import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt
import time


class Bandit:

    def __init__(self, params=None, cuda_available=False):
        # N: batch size; T: batch number; R: replication number; clip: clip probability
        if params is None:
            params = {'N': 5, 'T': 25, 'R': 1000, 'mean_reward': [0, 0], 'var_reward': [1, 1],
                      'clip': 0.1}
        self.N = params['N']
        self.T = params['T']
        self.R = params['R']
        self.clip = params['clip']
        self.num_act = 2
        if cuda_available:
            self.rand_temp = torch.zeros((self.R, self.N), dtype=torch.float).cuda()
            # temp variable for generating random variable
            self.crt_rwd = torch.zeros((self.R, self.N), dtype=torch.float).cuda()  # current reward
            self.crt_act = -torch.ones((self.R, self.N), dtype=torch.int).cuda()  # current action
        else:
            self.rand_temp = torch.zeros((self.R, self.N), dtype=torch.float)
            self.crt_rwd = torch.zeros((self.R, self.N), dtype=torch.float)
            self.crt_act = -torch.ones((self.R, self.N), dtype=torch.int)

    def normal_rwd(self):
        act_mean = torch.Tensor([0,0])
        self.crt_rwd[self.crt_act == 0] = self.rand_temp[self.crt_act == 0].normal_(0, 1)
        self.crt_rwd[self.crt_act == 1] = self.rand_temp[self.crt_act == 1].normal_(1, 1)

    def first_step(self):
        self.rand_temp.uniform_(to=self.num_act)
        self.crt_act = self.rand_temp.floor().type(torch.int)
        self.crt_act.sum(dim=1)
        self.crt_act[(self.crt_act.sum(dim=1) == 0), 0] = 1
        self.crt_act[(self.crt_act.sum(dim=1) == self.N), 0] = 0

    def cpt_rwd(self):
        # compute the reward given action.
        return


mbit = Bandit(cuda_available=True)
start = time.perf_counter()
mbit.first_step()
# for i in range(100):
#     np.random.normal(size=30000)
#     # torch.randn.cuda(100000)
#     # np.random.normal(size=100000)
end = time.perf_counter()
(mbit.crt_act.sum(dim=1)==0).sum()
print(end-start)
