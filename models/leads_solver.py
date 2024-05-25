"""
LEADS: Learning Dynamical Systems that Generalize Across Environments
"""
import pdb
import copy
import numpy as np

import torch
import torch.nn as nn
from torchdiffeq import odeint

from .submodules import CalculateNorm
from .vector_fields import *
from .abstract_solver import *
from .erm_solver import VectorField
from engine.configs import Algorithms


class EnvironVectorField(nn.Module):
    def __init__(self, state_dim, hidden_dim, data_name, n_env, is_ode, factor, nl, size=64, **kwargs):
        super().__init__()
        self.is_ode = is_ode
        self.n_env = n_env

        if self.is_ode:
            self.net_leaf = nn.ModuleList([GroupConvMLP(state_dim, state_dim, hidden_dim, groups=1, factor=factor, nl=nl) for _ in range(n_env)])
        elif data_name == "gray" or data_name == "wave":
            self.net_leaf = nn.ModuleList([GroupConv(state_dim, state_dim, hidden_dim, groups=1, factor=factor, nl=nl, size=size) for _ in range(n_env)])
        elif data_name == "navier":
            self.net_leaf = nn.ModuleList([GroupFNO2d(state_dim, state_dim, nl=nl, groups=1) for _ in range(n_env)])
        else:
            raise ValueError('Dataset {} is not supported!'.format(data_name))

    def forward(self, t, u):
        state = u.view(-1, self.n_env, *u.shape[1:])
        batch_size = state.size(0)
        if self.is_ode:
            state = state.permute(1, 0, 2)
            state = torch.split(state, 1, dim=0)
        else:
            state = torch.split(state, 1, dim=1)

        out = []
        for env_idx, env_state in enumerate(state):
            if self.is_ode:
                out.append(self.net_leaf[env_idx](env_state.squeeze(0)))
            else:
                out.append(self.net_leaf[env_idx](env_state.squeeze(1)))
        out = torch.stack(out, dim=0)

        if self.is_ode:
            out = out.permute(1, 0, 2)
            out = out.contiguous().view(batch_size * self.n_env, out.size(-1))
        else:
            out = torch.transpose(out, 0, 1)
            out = out.contiguous().view(batch_size*self.n_env, *out.shape[-3:])

        return out


@Algorithms.register('leads')
class LEADSSolver(AbstractSolver):
    def __init__(self, state_dim, hidden_dim, data_name, n_env, hparams, factor, options=None, nl="swish",
                 size=64, method='euler', is_ode=True, **kwargs):
        super().__init__(state_dim, hidden_dim, data_name, n_env, hparams)
        self.factor = factor
        self.options = options
        self.nl = nl
        self.method = method
        self.size = size
        self.int_ = odeint
        self.is_ode = is_ode
        self.kwargs = kwargs
        self.lambda_inv = self.hparams['lambda_inv']
        self.factor_lip = self.hparams['factor_lip']
        self._build()

    def _build(self):
        self.vector_field = VectorField(self.state_dim, self.hidden_dim, self.data_name, self.is_ode, self.factor,
                                        self.nl, self.size, **self.kwargs)
        self.vector_field_env = EnvironVectorField(self.state_dim, self.hidden_dim, self.data_name, self.n_env,
                                                   self.is_ode, self.factor, self.nl, self.size, **self.kwargs)
        self.loss_env_norm = CalculateNorm(self.vector_field_env.net_leaf)
        self.opt = self._get_optimizers(self.vector_field, self.vector_field_env)
        self.scheduler = self._get_scheduler(self.opt)
        self.criterion = self._get_loss_func()

    def _get_optimizers(self, model_func, model_func_ghost):
        opt_name = self.hparams['opt_name']

        if opt_name == 'adam':
            return torch.optim.Adam([{'params': model_func.parameters()},
                                     {'params': model_func_ghost.parameters(), 'lr': 1. * self.hparams['lr']}],
                                    lr=self.hparams['lr'], weight_decay=self.hparams['weight_decay'])
        elif opt_name == 'sgd':
            return torch.optim.SGD([{'params': model_func.parameters()},
                                    {'params': model_func_ghost.parameters(), 'lr': 1. * self.hparams['lr']}],
                                   lr=self.hparams['lr'], weight_decay=self.hparams['weight_decay'],
                                   momentum=self.hparams['momentum'], nesterov=True)
        else:
            raise Exception("Not support opt : {}".format(opt_name))

    def update(self, y, t, epsilon=0.99):
        self.opt.zero_grad()
        state = batch_transform_inverse(y, self.n_env)
        y_pred, y_pred_env = self.infer(y, t, epsilon, enable_deriv_min=True)
        loss = self.criterion(y_pred, state)

        loss_op_a = self.loss_env_norm.calculate_spectral_norm().sum()
        derivs = torch.split(y_pred_env, y.size(0))
        mini_batch_states = torch.split(state, y.size(0))

        loss_ops = [((deriv_e.norm(p=2, dim=1) / (state_e.norm(p=2, dim=1) + 1e-5)) ** 2).mean() for
                    deriv_e, state_e in zip(derivs, mini_batch_states)]

        loss_op_b = torch.stack(loss_ops).sum()
        loss_op = loss_op_a * self.factor_lip + loss_op_b

        if self.lambda_inv > 0:
            total_loss = loss + loss_op * self.lambda_inv
        else:
            total_loss = loss

        total_loss.backward()
        self.opt.step()

        return y_pred.permute(1, 2, 0) if self.is_ode else y_pred.permute(1, 2, 0, 3, 4), total_loss, loss, loss_op

    def infer(self, y, t, epsilon=0, enable_deriv_min=False):
        if epsilon < 1e-3:
            epsilon = 0

        state = batch_transform_inverse(y, self.n_env)
        state = state.permute(2, 0, 1) if self.is_ode else state.permute(2, 0, 1, 3, 4)

        y = y.permute(2, 0, 1) if self.is_ode else y.permute(2, 0, 1, 3, 4)

        if epsilon == 0:
            res = self.int_(self.vector_field, y0=state[0], t=t, method=self.method, options=self.options)
            res_env = self.int_(self.vector_field_env, y0=state[0], t=t, method=self.method, options=self.options)
            res = res + res_env
        else:
            eval_points = np.random.random(len(t)) < epsilon
            eval_points[-1] = False
            eval_points = eval_points[1:]
            start_i, end_i = 0, None
            res = []
            res_env = []
            for i, eval_point in enumerate(eval_points):
                if eval_point is True:
                    end_i = i + 1
                    t_seg = t[start_i:end_i + 1]
                    res_seg = self.int_(self.vector_field, y0=state[start_i], t=t_seg, method=self.method, options=self.options)
                    res_seg_env = self.int_(self.vector_field_env, y0=state[start_i], t=t_seg, method=self.method, options=self.options)
                    res_seg = res_seg + res_seg_env

                    if len(res) == 0:
                        res.append(res_seg), res_env.append(res_seg_env)
                    else:
                        res.append(res_seg[1:]), res_env.append(res_seg_env[1:])
                    start_i = end_i

            t_seg = t[start_i:]
            res_seg = self.int_(self.vector_field, y0=state[start_i], t=t_seg, method=self.method, options=self.options)
            res_seg_env = self.int_(self.vector_field_env, y0=state[start_i], t=t_seg, method=self.method, options=self.options)
            res_seg = res_seg + res_seg_env

            if len(res) == 0:
                res.append(res_seg), res_env.append(res_seg_env)
            else:
                res.append(res_seg[1:]), res_env.append(res_seg_env[1:])

            res = torch.cat(res, dim=0)
            res_env = torch.cat(res_env, dim=0)

        res = res.permute(1, 2, 0) if self.is_ode else res.permute(1, 2, 0, 3, 4)

        if enable_deriv_min:
            res_env = res_env.permute(1, 2, 0) if self.is_ode else res_env.permute(1, 2, 0, 3, 4)
            return res, res_env
        else:
            return res

    def adapt(self, y, t, epsilon=0.95):
        self.opt.zero_grad()
        targets = batch_transform_inverse(y, self.n_env)
        y_pred = self.infer(y, t, epsilon)
        loss = self.criterion(y_pred, targets)

        loss.backward()
        self.opt.step()
        return y_pred, loss, torch.tensor(0.), torch.tensor(0.)

    def set_no_grad(self):
        # Only training the g_e from scratch when adaptation
        params = []
        for name, param in self.vector_field_env.named_parameters():
            if param.requires_grad is False:
                param.requires_grad = True
            params.append(param)

            print(f"{name}, {param.requires_grad}")

        self.opt.param_groups[0]['params'] = params