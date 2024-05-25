"""
A general model for all environments
Created by Tiexin
"""

from torchdiffeq import odeint

from .vector_fields import *
from .abstract_solver import AbstractSolver
from engine.common import batch_transform_inverse
from engine.configs import Algorithms


class VectorField(nn.Module):
    def __init__(self, state_dim, hidden_dim, data_name, is_ode, factor, nl, size=64, **kwargs):
        super().__init__()
        self.is_ode = is_ode
        self.state_dim = state_dim

        if self.is_ode:
            self.net_root = GroupConvMLP(state_dim, state_dim, hidden_dim, groups=1, factor=factor, nl=nl)
        elif data_name == "gray" or data_name == "wave":
            self.net_root = GroupConv(state_dim, state_dim, hidden_dim, groups=1, factor=factor, nl=nl, size=size)
        elif data_name == "navier":
            self.net_root = GroupFNO2d(state_dim, state_dim, groups=1, nl=nl)

    def forward(self, t, u):
        if self.net_root.__class__.__name__ in ['GroupConvMLP']:
            u = u.view(u.size(0), -1, self.state_dim)
            mini_batch_size = u.size(0)
            out = self.net_root(u.view(-1, self.state_dim))
            out = out.view(mini_batch_size, -1, self.state_dim)
            out = out.view(mini_batch_size, out.size(1) * self.state_dim)
        else:
            out = self.net_root(u)

        return out


@Algorithms.register('erm')
class ERMSolver(AbstractSolver):
    def __init__(self, state_dim, hidden_dim, data_name, n_env, hparams, factor, nl="swish", size=64, method='euler', is_ode=True):
        super().__init__(state_dim, hidden_dim, data_name, n_env, hparams)
        self.factor = factor
        self.nl = nl
        self.size = size
        self.method = method
        self.is_ode = is_ode
        self.int_ = odeint
        self.options = {}
        self._build()

    def _build(self):
        self.vector_field = VectorField(self.state_dim, self.hidden_dim, self.data_name, self.is_ode, self.factor, self.nl, self.size)
        self.opt = self._get_optimizer(self.vector_field)
        self.scheduler = self._get_scheduler(self.opt)
        self.criterion = self._get_loss_func()

    def _infer(self, y, t):
        """
        :param y: [batch_size, n_env * state_dim, time_steps]
        :param t: [time_steps]
        :return:
        """
        state = batch_transform_inverse(y, self.n_env)
        state = state.permute(2, 0, 1) if self.is_ode else state.permute(2, 0, 1, 3, 4) # [20, 36, 2]
        y_pred = self.int_(self.vector_field, y0=state[0], t=t, method=self.method, options=self.options)
        return y_pred.permute(1, 2, 0) if self.is_ode else y_pred.permute(1, 2, 0, 3, 4)

    def update(self, y, t):
        self.opt.zero_grad()
        targets = batch_transform_inverse(y, self.n_env) # [288, 2, 100]
        y_pred = self._infer(y, t)
        loss = self.criterion(y_pred, targets)
        loss.backward()
        self.opt.step()
        return y_pred, loss, loss, loss

    def infer(self, y, t):
        return self._infer(y, t)

    def adapt(self, y, t):
        return
