"""
https://github.com/yuan-yin/CoDA
"""
import abc
from abc import ABC

import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch.utils.data import Dataset
from functools import partial


class ODEDataset(Dataset, ABC):
    def __init__(self, n_data_per_env, t_horizon, params, dt, random_influence=0.2, method='RK45', split='train', rdn_gen=1.):
        """
        Abstract dataset for ODE dynamics
        :param n_data_per_env:
        :param t_horizon:
        :param params:
        :param dt:
        :param random_influence:
        :param method:
        :param split: train/val/test
        :param rdn_gen: shift value for initial condition
        """
        super().__init__()
        self.n_data_per_env = n_data_per_env
        self.num_env = len(params)
        self.len = n_data_per_env * self.num_env
        self.t_horizon = t_horizon
        self.dt = dt
        self.random_influence = random_influence
        self.params_eq = params
        self.split = split
        self.method = method
        self.rdn_gen = rdn_gen
        self.max = np.iinfo(np.int32).max
        self.buffer = dict()
        self.indices = [list(range(env * n_data_per_env, (env + 1) * n_data_per_env)) for env in range(self.num_env)]

    @abc.abstractmethod
    def _f(self, t, x, env):
        pass

    @abc.abstractmethod
    def _get_init_cond(self, index):
        pass

    def __getitem__(self, index):
        env = index // self.n_data_per_env
        env_index = index % self.n_data_per_env
        t = torch.arange(0, self.t_horizon, self.dt).float()
        out = {'t': t,
               'env': env}
        if self.buffer.get(index) is None:
            y0 = self._get_init_cond(env_index)
            y = solve_ivp(partial(self._f, env=env), (0., self.t_horizon), y0=y0, method=self.method,
                          t_eval=np.arange(0., self.t_horizon, self.dt))
            y = torch.from_numpy(y.y).float()
            out['state'] = y
            self.buffer[index] = y.numpy()
        else:
            out['state'] = torch.from_numpy(self.buffer[index])

        out['index'] = index
        out['param'] = torch.tensor(list(self.params_eq[env].values()))

        return out

    def __len__(self):
        return self.len
