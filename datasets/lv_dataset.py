###################################################################
# Lotka-Volterra
###################################################################

import numpy as np
from .ode_dataset import ODEDataset
from engine.configs import Datasets


@Datasets.register('lotka')
class LotkaVolterraDataset(ODEDataset):
    def _f(self, t, x, env=0):
        alpha = self.params_eq[env]['alpha']
        beta = self.params_eq[env]['beta']
        gamma = self.params_eq[env]['gamma']
        delta = self.params_eq[env]['delta']
        d = np.zeros(2)
        d[0] = alpha * x[0] - beta * x[0] * x[1]
        d[1] = delta * x[0] * x[1] - gamma * x[1]
        return d

    def _get_init_cond(self, index):
        # test environments use different initial condition from training environments
        # print('Mode {} || Env:{}, seed:{}'.format(self.split, index, index if not self.split == 'test' else self.max - index))
        np.random.seed(index if not self.split == 'test' else self.max - index)
        return np.random.random(2) + self.rdn_gen
