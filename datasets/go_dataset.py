###################################################################
# Glycolitic Oscillator
###################################################################
import numpy as np
from .ode_dataset import ODEDataset
from engine.configs import Datasets


@Datasets.register('g_osci')
class GlycolyticOscillatorDataset(ODEDataset):
    def _f(self, t, x, env=0):
        keys = ['J0', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'K1', 'q', 'N', 'A', 'kappa', 'psi', 'k']
        J0, k1, k2, k3, k4, k5, k6, K1, q, N, A, kappa, psi, k = [self.params_eq[env][k] for k in keys]

        d = np.zeros(7)
        k1s1s6 = k1 * x[0] * x[5] / (1 + (x[5]/K1) ** q)
        d[0] = J0 - k1s1s6
        d[1] = 2 * k1s1s6 - k2 * x[1] * (N - x[4]) - k6 * x[1] * x[4]
        d[2] = k2 * x[1] * (N - x[4]) - k3 * x[2] * (A - x[5])
        d[3] = k3 * x[2] * (A - x[5]) - k4 * x[3] * x[4] - kappa * (x[3] - x[6])
        d[4] = k2 * x[1] * (N - x[4]) - k4 * x[3] * x[4] - k6 * x[1] * x[4]
        d[5] = -2 * k1s1s6 + 2 * k3 * x[2] * (A - x[5]) - k5 * x[5]
        d[6] = psi * kappa * (x[3] - x[6]) - k * x[6]
        return d

    def _get_init_cond(self, index):
        # test environments use different initial condition from training environments
        np.random.seed(index if not self.split == 'test' else self.max - index)
        ic_range = [(0.15, 1.60), (0.19, 2.16), (0.04, 0.20), (0.10, 0.35), (0.08, 0.30), (0.14, 2.67), (0.05, 0.10)]
        return np.random.random(7) * np.array([b-a for a, b in ic_range]) + np.array([a for a, _ in ic_range])
