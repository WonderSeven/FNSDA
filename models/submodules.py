import pdb
import copy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from engine.common import set_requires_grad


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))
    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)

class Sinus(nn.Module):
    def forward(self, input):
        return torch.sinus(input)

class GroupSwish(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5 for _ in range(groups)]))
        self.groups = groups

    def forward(self, x):
        n_ch_group = x.size(1) // self.groups
        t = x.shape[2:]
        x = x.reshape(-1, self.groups, n_ch_group, *t)
        beta = self.beta.view(1, self.groups, 1, *[1 for _ in t])
        return (x * torch.sigmoid_(x * F.softplus(beta))).div_(1.1).reshape(-1, self.groups * n_ch_group, *t)


nls = {'relu': partial(nn.ReLU),
       'sigmoid': partial(nn.Sigmoid),
       'tanh': partial(nn.Tanh),
       'selu': partial(nn.SELU),
       'softplus': partial(nn.Softplus),
       'swish': partial(Swish),
       'sinus': partial(Sinus),
       'group_swish': partial(GroupSwish),
       'elu': partial(nn.ELU)}


class GroupActivation(nn.Module):
    def __init__(self, nl, groups=1):
        super().__init__()
        self.groups = groups
        if nl == 'group_swish':
            self.activation = nls[nl](groups)
        else:
            self.activation = nls[nl]()

    def forward(self, x):
        return self.activation(x)

################################################################################
# Modules for CoDA https://github.com/yuan-yin/CoDA/blob/main/network.py
################################################################################
class HyperEnvNet(nn.Module):
    def __init__(self, net_a, ghost_structure, hypernet, codes, net_mask=None):
        super().__init__()
        self.net_a = net_a
        self.codes = codes
        self.n_env = codes.size(0)
        self.hypernet = hypernet
        self.nets = {'ghost_structure': ghost_structure, "mask": net_mask}  # , "ghost": ghost_structure}

    def update_ghost(self):
        net_ghost = copy.deepcopy(self.nets['ghost_structure'])
        set_requires_grad(net_ghost, False)
        self.nets["ghost"] = net_ghost
        param_hyper = self.hypernet(self.codes)
        count_f = 0
        count_p = 0
        param_mask = self.nets["mask"]

        for pa, pg in zip(self.net_a.parameters(), self.nets["ghost"].parameters()):
            phypers = []
            if param_mask is None:
                pmask_sum = int(pa.numel())
            else:
                pmask = param_mask[count_f: count_f + pa.numel()].reshape(*pa.shape)
                pmask_sum = int(pmask.sum())
            if pmask_sum == int(pa.numel()):
                for e in range(self.n_env):
                    phypers.append(param_hyper[e, count_p: count_p + pmask_sum].reshape(*pa.shape))
            else:
                for e in range(self.n_env):
                    phyper = torch.zeros(*pa.shape).cuda()
                    if pmask_sum != 0:
                        phyper[pmask == 1] = param_hyper[count_p:count_p + pmask_sum]
                    phypers.append(phyper)
            count_p += pmask_sum
            count_f += int(pa.numel())

            phyper = torch.cat(phypers, dim=0)
            pa_new = torch.cat([pa] * self.n_env, dim=0)
            pg.copy_(pa_new + phyper)

    def forward(self, *input, **kwargs):
        return self.nets["ghost"](*input, **kwargs)


################################################################################
# Modules for LEADS https://github.com/yuan-yin/LEADS/blob/main/utils.py#L104
################################################################################
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class CalculateNorm(nn.Module):
    def __init__(self, module, power_iterations=5, use_cuda=True):
        super().__init__()
        self.module = module
        assert isinstance(module, nn.ModuleList)
        self.power_iterations = power_iterations
        self.use_cuda = use_cuda
        self._make_params()

    def calculate_spectral_norm(self):
        # Adapted to complex weights
        sigmas = [0. for i in range(len(self.module))]
        for i, module in enumerate(self.module):
            for name, w in module.named_parameters():
                if name.find('bias') == -1 and name.find('beta') == -1:
                    u = self.u[f'{i},{name}']
                    v = self.v[f'{i},{name}']
                    height = w.data.shape[0]

                    for _ in range(self.power_iterations):
                        v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
                        u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

                    sigma = torch.conj(u).dot(w.view(height, -1).mv(v))
                    if torch.is_complex(sigma):
                        sigmas[i] = sigmas[i] + sigma.real ** 2
                    else:
                        sigmas[i] = sigmas[i] + sigma ** 2
        return torch.stack(sigmas)

    def calculate_frobenius_norm(self):
        # Only used for linear case
        sigmas = [0. for i in range(len(self.module))]
        for i, module in enumerate(self.module):
            for name, w in module.named_parameters():
                if name.find('bias') == -1 and name.find('beta') == -1:
                    sigmas[i] = sigmas[i] + torch.norm(w)
        return torch.stack(sigmas)

    def _make_params(self):
        self.u, self.v = dict(), dict()
        for i, module in enumerate(self.module):
            for name, w in module.named_parameters():
                if name.find('bias') == -1 and name.find('beta') == -1:
                    height = w.data.shape[0]
                    width = w.view(height, -1).data.shape[1]

                    u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
                    v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)

                    if self.use_cuda:
                        u, v = u.cuda(), v.cuda()

                    u.data = l2normalize(u.data)
                    v.data = l2normalize(v.data)

                    self.u[f'{i},{name}'] = u
                    self.v[f'{i},{name}'] = v

################################################################################
# Modules for Fourier layer
################################################################################
class GroupSpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, groups=1):
        super(GroupSpectralConv1d, self).__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.scale = 1 / (in_channels * out_channels)
        self.groups = groups
        self.weights = nn.Parameter(self.scale * torch.rand(groups * self.in_channels, out_channels, self.modes1, 2))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, env, in_channel, x), (in_channel, out_channel, x) -> (batch, env, out_channel, x)
        return torch.einsum("beix,iox->beox", input, weights)

    def forward(self, x):
        """
        :param x: [100, 9, 20]
        :return:
        """
        batch_size, n_env = x.shape[0], x.shape[1]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
        x_ft = x_ft.reshape(batch_size, n_env, self.in_channels, x.size(-1) // 2 + 1)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, n_env, self.out_channels, x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device) # [100, 9, 1, 11]
        out_ft[:, :, :, : self.modes1] = self.compl_mul1d(x_ft[:, :, :, :self.modes1], torch.view_as_complex(self.weights))

        # Return to physical space
        out_ft = out_ft.reshape(batch_size, n_env * self.out_channels, x.size(-1) // 2 + 1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class GroupSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, groups=1):
        super().__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.groups = groups
        self.weights1 = nn.Parameter(self.scale * torch.rand(groups * in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(groups * in_channels, out_channels, self.modes1, self.modes2, 2))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, env, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, env, out_channel, x,y)
        return torch.einsum("beixy,eioxy->beoxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        x_ft = x_ft.reshape(batchsize, self.groups, self.in_channels, x.size(-2), x.size(-1) // 2 + 1)

        # Multiply relevant Fourier modes
        weights1 = self.weights1.reshape(self.groups, self.in_channels, self.out_channels, self.modes1, self.modes2, 2)
        weights2 = self.weights2.reshape(self.groups, self.in_channels, self.out_channels, self.modes1, self.modes2, 2)
        out_ft = torch.zeros(batchsize, self.groups, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :, :self.modes1, :self.modes2], torch.view_as_complex(weights1))
        out_ft[:, :, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, :, -self.modes1:, :self.modes2], torch.view_as_complex(weights2))

        # Return to physical space
        out_ft = out_ft.reshape(batchsize, self.groups * self.out_channels, x.size(-2), x.size(-1) // 2 + 1)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        return x
