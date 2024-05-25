"""
First-order Context-based Adaptation for Generalizing to New Dynamical Systems.
"""

from torchdiffeq import odeint

from engine.common import *
from .vector_fields import *
from .abstract_solver import *
from engine.configs import Algorithms


class VectorField(nn.Module):
    def __init__(self, state_dim, hidden_dim, code_dim, data_name, n_env, is_ode, factor, nl, size=64, tau=0.1, **kwargs):
        super().__init__()
        self.is_ode = is_ode
        self.n_env = n_env
        self.tau = tau
        if is_ode:
            self.codes = nn.Parameter(torch.zeros(n_env, code_dim))
        else:
            self.codes = nn.Parameter(torch.zeros(n_env, state_dim, 32, 32))

        # Bias
        if self.is_ode:
            self.net_root = GroupConvMLP(state_dim+code_dim, state_dim, hidden_dim, groups=1, factor=factor, nl=nl)
        elif data_name == "gray" or data_name == "wave":
            self.net_root = GroupConv(state_dim * 2, state_dim, hidden_dim, groups=1, factor=factor, nl=nl, size=size)
        elif data_name == "navier":
            self.net_root = GroupFNO2d(state_dim * 2, state_dim, nl=nl, groups=1)

        # Ghost
        self.net_leaf = copy.deepcopy(self.net_root)
        set_requires_grad(self.net_leaf, False)

        self.net_combined = None

    def ema_update(self):
        """
        EMA update
        :return:
        """
        for param_root, param_leaf in zip(self.net_root.parameters(), self.net_leaf.parameters()):
            param_leaf.copy_(self.tau * param_root + (1 - self.tau) * param_leaf)

    def update_root(self):
        self.net_combined = self.net_root

    def update_leaf(self):
        self.net_combined = self.net_leaf

    def update_code(self, value):
        self.codes.data.copy_(value)

    def forward(self, t, u):
        batch_size = u.size(0)
        codes = self.codes.unsqueeze(0)
        input_dim = u.ndimension()
        if input_dim == 2:
            u = u.view(batch_size, self.n_env, -1)
            codes = codes.repeat([batch_size, 1, 1])
            stacked_state = torch.cat([u, codes], dim=-1)
            stacked_state = stacked_state.view(batch_size * self.n_env, -1)
        elif input_dim == 4:
            u = u.view(batch_size, self.n_env, u.shape[1] //self.n_env , *u.shape[-2:])
            codes = codes.repeat([batch_size, 1, 1, 1, 1])
            stacked_state = torch.cat([u, codes], dim=2)
            stacked_state = stacked_state.view(batch_size * self.n_env, *stacked_state.shape[2:])
        else:
            raise ValueError

        out = self.net_combined(stacked_state)

        if input_dim == 2:
            out = out.view(batch_size, self.n_env, out.size(-1))
            out = out.view(batch_size, self.n_env * out.size(-1))
        elif input_dim == 4:
            out = out.view(batch_size, self.n_env * out.shape[1], *out.shape[-2:])
        else:
            raise ValueError

        return out


@Algorithms.register('foca')
class FOCASolver(AbstractSolver):
    def __init__(self, state_dim, hidden_dim, code_dim, data_name, n_env, hparams, factor, options=None, nl="swish",
                 size=64, method='euler', is_ode=True, **kwargs):
        super().__init__(state_dim, hidden_dim, data_name, n_env, hparams)
        self.code_dim = code_dim
        self.factor = factor
        self.nl = nl
        self.size = size
        self.is_ode = is_ode
        self.options = dict(options)
        self.method = method
        self.int_ = odeint
        self.tau = self.hparams['tau']
        self.code_lr = self.hparams['code_lr']
        self.code_steps = self.hparams['code_steps']
        self.kwargs = kwargs
        self._build()

    def _build(self):
        self.vector_field = VectorField(self.state_dim, self.hidden_dim, self.code_dim, self.data_name, self.n_env,
                                        self.is_ode, self.factor, self.nl, self.size, self.tau, **self.kwargs)
        self.opt = self._get_optimizer(self.vector_field.net_root)
        self.scheduler = self._get_scheduler(self.opt)
        self.criterion = self._get_loss_func()

    def _infer(self, y, t, epsilon=0):
        if epsilon < 1e-3:
            epsilon = 0

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

    def update(self, y, t, epsilon=0.99):
        self.opt.zero_grad()
        targets = batch_transform_inverse(y, self.n_env)

        # Step 1: Update codes with fixed model params
        self.vector_field.update_leaf()
        codes_loss = []
        for k in range(self.code_steps):
            y_pred = self._infer(y, t, epsilon)
            loss = self.criterion(y_pred, targets)
            partial_loss_codes = torch.autograd.grad(loss, self.vector_field.codes, grad_outputs=torch.ones_like(loss),
                                                     retain_graph=True)[0]
            self.vector_field.update_code(self.vector_field.codes - self.code_lr * partial_loss_codes)
            codes_loss.append(loss.detach())
        codes_loss = torch.stack(codes_loss)

        # Step 2: Update model params with fixed codes
        self.vector_field.update_root()
        y_pred = self._infer(y, t, epsilon)
        loss = self.criterion(y_pred, targets)
        loss.backward()
        self.opt.step()
        self.scheduler.step()

        # Step 3: ema update
        self.vector_field.ema_update()

        return y_pred, loss, loss, torch.mean(codes_loss)

    def infer(self, y, t, epsilon=0):
        self.vector_field.update_root()
        return self._infer(y, t, epsilon)

    def adapt(self, y, t, epsilon=0.95):
        return self.update(y, t, epsilon)

    def set_no_grad(self):
        pass
