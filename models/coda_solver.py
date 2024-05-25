"""
Generalizing to New Physical Systems via Context-Informed Dynamics Model
"""


import torch
import torch.nn as nn
from torchdiffeq import odeint

from engine.common import *
from .vector_fields import *
from .submodules import HyperEnvNet
from .abstract_solver import *
from engine.configs import Algorithms


class VectorField(nn.Module):
    def __init__(self, state_dim, hidden_dim, code_dim, data_name, n_env, is_ode, factor, nl, size=64, is_layer=False,
                 layers=[0], mask=None, codes_init=None, **kwargs):
        super().__init__()
        self.is_ode = is_ode
        self.size = size
        self.is_layer = is_layer
        self.codes = nn.Parameter(0. * torch.ones(n_env, code_dim)) if codes_init is None else codes_init
        # Bias
        if self.is_ode:
            self.net_root = GroupConvMLP(state_dim, state_dim, hidden_dim, groups=1, factor=factor, nl=nl)
        elif data_name == "gray" or data_name == "wave":
            self.net_root = GroupConv(state_dim, state_dim, hidden_dim, groups=1, factor=factor, nl=nl, size=size)
        elif data_name == "navier":
            self.net_root = GroupFNO2d(state_dim, state_dim, nl=nl, groups=1)
        n_param_tot = count_parameters(self.net_root)
        n_param_mask = n_param_tot if not is_layer else get_n_param_layer(self.net_root, layers)
        n_param_hypernet = n_param_mask
        print(f"Params: n_mask {n_param_mask} / n_tot {n_param_tot} / n_hypernet {n_param_hypernet}")
        self.net_hyper = nn.Linear(code_dim, n_param_hypernet, bias=False)

        # Ghost
        if self.is_ode:
            self.ghost_structure = GroupConvMLP(state_dim, state_dim, hidden_dim, groups=n_env, factor=factor, nl=nl)
        elif data_name == "gray" or data_name == "wave":
            self.ghost_structure = GroupConv(state_dim, state_dim, hidden_dim, groups=n_env, factor=factor, nl=nl, size=size)
        elif data_name == "navier":
            self.ghost_structure = GroupFNO2d(state_dim, state_dim, nl=nl, groups=n_env)
        else:
            raise Exception(f"{data_name} net not implemented")

        set_requires_grad(self.ghost_structure, False)

        # Mask
        if is_layer and mask is None:
            self.mask = {"mask": generate_mask(self.net_root, "layer", layers)}
        else:
            self.mask = {"mask": mask}

        self.net_leaf = HyperEnvNet(self.net_root, self.ghost_structure, self.net_hyper, self.codes, self.mask["mask"], **kwargs)

    def update_ghost(self):
        self.net_leaf.update_ghost()

    def forward(self, t, u):
        return self.net_leaf(u)


@Algorithms.register('coda')
class CoDASolver(AbstractSolver):
    def __init__(self, state_dim, hidden_dim, code_dim, data_name, n_env, hparams, factor, options=None, nl="swish",
                 size=64, method='euler', is_layer=False, is_ode=True, layers=[0], mask=None, codes_init=None, **kwargs):
        super().__init__(state_dim, hidden_dim, data_name, n_env, hparams)
        self.code_dim = code_dim
        self.factor = factor
        self.nl = nl
        self.size = size
        self.is_ode = is_ode
        self.options = dict(options)
        self.method = method
        self.is_layer = is_layer
        self.layers = layers
        self.mask = mask
        self.codes_init = codes_init
        self.int_ = odeint
        self.loss_norm = self.hparams['loss_norm'] # 'l12m-l1c'  # 'l2m-l2c'
        self.l_m = float(self.hparams['l_m'])
        self.l_c = self.l_m * 100
        self.l_t = self.l_m
        self.kwargs = kwargs
        self._build()

    def _build(self):
        self.vector_field = VectorField(self.state_dim, self.hidden_dim, self.code_dim, self.data_name, self.n_env,
                                        self.is_ode, self.factor, self.nl, self.size, self.is_layer, self.layers,
                                        self.mask, self.codes_init, **self.kwargs)
        self.opt = self._get_optimizer(self.vector_field)
        self.scheduler = self._get_scheduler(self.opt)
        self.criterion = self._get_loss_func()

    def update(self, y, t, epsilon=0.99):
        # Regularization
        loss_reg_row = torch.zeros(1).to(y.device)
        loss_reg_col = torch.zeros(1).to(y.device)
        loss_reg_theta = torch.zeros(1).to(y.device)
        loss_reg_code = torch.zeros(1).to(y.device)

        self.opt.zero_grad()
        targets = batch_transform_inverse(y, self.n_env)
        y_pred = self.infer(y, t, epsilon)
        loss = self.criterion(y_pred, targets)

        if "l1c" in self.loss_norm:
            # L1 norm for the contexts
            loss_reg_code += torch.norm(self.vector_field.codes, p=1, dim=0).sum()
        if "l2c" in self.loss_norm:
            # L2 norm for the contexts
            loss_reg_code += (torch.norm(self.vector_field.codes, dim=0) ** 2).sum()
        if "l12m" in self.loss_norm:
            # L1-L2 norm for the HyperNet
            loss_reg_row += (torch.norm(self.vector_field.net_hyper.weight, dim=1)).sum()
        if "l2m" in self.loss_norm:
            # L2 norm for the HyperNet
            loss_reg_row += (torch.norm(self.vector_field.net_hyper.weight, dim=1) ** 2).sum()  # [232421, 2] [125740.1484]

        loss = loss + self.l_m * (loss_reg_row + loss_reg_col) + self.l_t * loss_reg_theta + self.l_c * loss_reg_code

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vector_field.parameters(), 1.)
        self.opt.step()

        return y_pred, loss, torch.tensor(0.), torch.tensor(0.)

    def infer(self, y, t, epsilon=0):
        """
        :param y: [4, 18, 20]  NS:[50, 5, 10, 32, 32]
        :param t:
        :param epsilon:
        :return:
        """
        if epsilon < 1e-3:
            epsilon = 0

        self.vector_field.update_ghost()

        y = y.permute(2, 0, 1) if self.is_ode else y.permute(2, 0, 1, 3, 4) # [20, 36, 2]

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
                    res_seg = self.int_(self.vector_field, y0=y[start_i], t=t_seg,
                                        method=self.method, options=self.options)
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

    def adapt(self, y, t, epsilon=0.95):
        self.opt.zero_grad()
        targets = batch_transform_inverse(y, self.n_env)
        y_pred = self.infer(y, t, epsilon)
        loss = self.criterion(y_pred, targets)

        loss.backward()
        self.opt.step()
        return y_pred, loss, torch.tensor(0.), torch.tensor(0.)

    def set_no_grad(self):
        params = []
        for idx, (name, param) in enumerate(self.vector_field.named_parameters()):
            if param.requires_grad and ("net_root" in name or "net_hyper" in name or "mask" in name):
                param.requires_grad = False
            else:
                params.append(param)

        self.opt.param_groups[0]['params'] = params
