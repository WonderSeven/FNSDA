"""
First-order Context-based Adaptation for Generalizing to New Dynamical Systems.
"""
import pdb
import numpy as np

import torch
import torch.nn as nn

from torchdiffeq import odeint
from .abstract_solver import AbstractSolver
from .fourier_submodules import GroupFNO1d, GrayGroupFNO2d, GroupFNO2d
from engine.common import batch_transform_inverse
from engine.configs import Algorithms


class VectorField(nn.Module):
    def __init__(self, state_dim, hidden_dim, code_dim, data_name, n_env, is_ode, factor, nl, size=64, **kwargs):
        super(VectorField, self).__init__()
        self.is_ode = is_ode
        self.n_env = n_env
        self.codes = nn.Parameter(torch.zeros(n_env, code_dim), requires_grad=True)

        # Bias
        if self.is_ode:
            self.net_root = GroupFNO1d(state_dim, state_dim, code_dim, n_env=n_env, nl=nl, groups=10)
        elif data_name == "gray" or data_name == "wave":
            self.net_root = GrayGroupFNO2d(state_dim, state_dim, code_dim, n_env=n_env, nl=nl, groups=1)
        elif data_name == "navier":
            self.net_root = GroupFNO2d(state_dim, state_dim, code_dim, n_env=n_env, nl=nl, groups=1)

        self.net_combined = None

    def update_code(self, value):
        self.codes.data.copy_(value)

    def get_env_code_weights(self):
        return self.net_root.get_env_code_weights()

    def forward(self, t, u):
        return self.net_root(u, self.codes)


@Algorithms.register('fourier')
class FourierSolver(AbstractSolver):
    def __init__(self, state_dim, hidden_dim, code_dim, data_name, n_env, hparams, factor, options=None, nl="swish",
                 size=64, method='euler', is_ode=True, **kwargs):
        super(FourierSolver, self).__init__(state_dim, hidden_dim, data_name, n_env, hparams)
        self.code_dim = code_dim
        self.factor = factor
        self.nl = nl
        self.size = size
        self.is_ode = is_ode
        self.options = dict(options)
        self.method = method
        self.int_ = odeint
        self.loss_norm = self.hparams['loss_norm'] # l1c l2c
        self.l_m = float(self.hparams['l_m'])
        self.l_f = self.l_m
        self.l_c = float(self.hparams['l_c'])
        self.kwargs = kwargs
        self._build()

    def _build(self):
        self.vector_field = VectorField(self.state_dim, self.hidden_dim, self.code_dim, self.data_name, self.n_env,
                                        self.is_ode, self.factor, self.nl, self.size, **self.kwargs)
        self.opt = self._get_optimizer(self.vector_field)
        self.scheduler = self._get_scheduler(self.opt)
        self.criterion = self._get_loss_func()

    def _infer(self, y, t, epsilon=0):
        if epsilon < 1e-3:
            epsilon = 0

        y = y.permute(2, 0, 1) if self.is_ode else y.permute(2, 0, 1, 3, 4)

        if epsilon == 0:
            res = self.int_(self.vector_field, y0=y[0], t=t, method=self.method, options=self.options)
        else:
            eval_points = np.random.random(len(t)) < epsilon
            eval_points[-1] = False
            eval_points = eval_points[1:]
            start_i, end_i = 0, None
            res = []
            for i, eval_point in enumerate(eval_points):
                if eval_point is True:
                    end_i = i + 1
                    t_seg = t[start_i:end_i + 1]
                    res_seg = self.int_(self.vector_field, y0=y[start_i], t=t_seg, method=self.method, options=self.options)
                    if len(res) == 0:
                        res.append(res_seg)
                    else:
                        res.append(res_seg[1:])
                    start_i = end_i
            t_seg = t[start_i:]
            res_seg = self.int_(self.vector_field, y0=y[start_i], t=t_seg, method=self.method, options=self.options)
            if len(res) == 0:
                res.append(res_seg)
            else:
                res.append(res_seg[1:])
            res = torch.cat(res, dim=0)
        res = res.permute(1, 2, 0) if self.is_ode else res.permute(1, 2, 0, 3, 4)
        return batch_transform_inverse(res, self.n_env)

    def update(self, y, t, epsilon=0.99):
        loss_reg_row = torch.zeros(1).to(y.device)
        loss_reg_code = torch.zeros(1).to(y.device)
        self.opt.zero_grad()

        targets = batch_transform_inverse(y, self.n_env)
        y_pred = self._infer(y, t, epsilon)
        loss = self.criterion(y_pred, targets)

        if "l1c" in self.loss_norm:
            # L1 norm for the contexts
            loss_reg_code += torch.norm(self.vector_field.codes, p=1, dim=0).sum()
        if "l2c" in self.loss_norm:
            # L2 norm for the contexts
            loss_reg_code += (torch.norm(self.vector_field.codes, dim=0) ** 2).sum()
        if "l12m" in self.loss_norm:
            # L1-L2 norm for the weight W_e
            for (k, v) in self.vector_field.named_parameters():
                if 'code_weights' in k:
                    loss_reg_row += (torch.norm(v, dim=-1)).sum()
        if "l2m" in self.loss_norm:
            # L2 norm for the weight W_e
            for (k, v) in self.vector_field.named_parameters():
                if 'code_weights' in k:
                    loss_reg_row += (torch.norm(v, dim=-1) ** 2).sum()

        total_loss = loss + self.l_c * loss_reg_code + self.l_m * loss_reg_row

        total_loss.backward()
        self.opt.step()
        self.scheduler.step()

        return y_pred, total_loss, loss, loss_reg_code

    def infer(self, y, t, epsilon=0):
        return self._infer(y, t, epsilon)

    def adapt(self, y, t, epsilon=0.95):
        return self.update(y, t, epsilon)

    def set_no_grad(self):
        params = []
        for idx, (name, param) in enumerate(self.vector_field.named_parameters()):
            if ("codes" in name or "activation" in name):
                param.requires_grad = True
                params.append(param)
            else:
                param.requires_grad = False

            print(f"{name}, {param.requires_grad}")

        self.opt.param_groups[0]['params'] = params
