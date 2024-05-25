import os
import pdb

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from .submodules import GroupActivation


class GroupSpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, code_channels, n_env, modes1):
        super(GroupSpectralConv1d, self).__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        Using learnable splitter.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.code_channels = code_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.scale = 1 / (in_channels * out_channels)
        self.n_env = n_env
        self.com_weights = nn.Parameter(self.scale * torch.rand(self.in_channels, out_channels, self.modes1, 2))
        self.code_weights = nn.Parameter(self.scale * torch.rand(code_channels * in_channels, out_channels, self.modes1, 2))
        self.filter_weights = nn.Parameter(-3. * torch.ones(self.modes1, dtype=torch.float32), requires_grad=True)

    def compl_mul1d(self, inputs, com_weights, env_weights, lam):
        return (lam * torch.einsum("beix,iox->beox", inputs, torch.view_as_complex(com_weights)) +
                (1. - lam) * torch.einsum("beix,eiox->beox", inputs, torch.view_as_complex(env_weights)))

    def forward(self, x, codes):
        """
        :param x: [4, 9, 20]
        :param codes: [9, 20]
        :return:
        """
        batch_size = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
        x_ft = x_ft.reshape(batch_size, self.n_env, self.in_channels, x.size(-1) // 2 + 1)

        out_ft = torch.zeros(batch_size, self.n_env, self.out_channels, x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        filter_lam = F.hardsigmoid(self.filter_weights)

        com_weights = self.com_weights.reshape(self.in_channels, self.out_channels, self.modes1, 2)
        code_weights = self.code_weights.reshape(self.code_channels, self.in_channels, self.out_channels, self.modes1, 2)
        env_weights = torch.einsum("ec,cioxy->eioxy", codes, code_weights)
        out_ft[:, :, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :, :self.modes1], com_weights, env_weights, filter_lam)

        # Return to physical space
        out_ft = out_ft.reshape(batch_size, self.n_env * self.out_channels, x.size(-1) // 2 + 1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


# ---------------------------------------------- ODE ---------------------------------------------------------------
class GroupFNO1d(nn.Module):
    def __init__(self, input_dim, output_dim, code_dim, n_env=1, modes1=5, width=10, groups=1, nl='swish'):
        super().__init__()
        self.width = width
        self.n_env = n_env
        self.groups = groups

        self.p = nn.Conv1d(2 * input_dim, 2 * width, kernel_size=1)
        self.conv0 = GroupSpectralConv1d(1, 1, code_dim, n_env, modes1)
        self.conv1 = GroupSpectralConv1d(1, 1, code_dim, n_env, modes1)
        self.w0 = nn.Conv1d(2 * width, 2 * width, 1, groups=groups)
        self.w1 = nn.Conv1d(2 * width, 2 * width, 1, groups=groups)
        self.a0 = GroupActivation(nl, groups=n_env)
        self.a1 = GroupActivation(nl, groups=n_env)
        self.fc1 = nn.Conv1d(2 * width, (output_dim // 2) * width, 1, groups=width)
        self.fc2 = nn.Conv1d((output_dim // 2) * width, output_dim, kernel_size=1)

    def forward(self, x, codes):
        x = x.view(x.size(0), self.n_env, -1)
        mini_batch_size, n_env, size_x = x.size()
        grid = self.get_grid(mini_batch_size, n_env, size_x, x.device)
        x = torch.cat([x, grid], dim=-1)
        x = x.view(mini_batch_size * n_env, -1, 1)

        # Lift with P
        x = self.p(x)
        x = x.view(mini_batch_size, n_env, -1)
        # Fourier layer 0
        x1 = self.conv0(x, codes)
        x2 = self.w0(x.view(mini_batch_size * n_env, -1, 1))
        x = self.a0(x1 + x2.view(mini_batch_size, n_env, -1))

        # Fourier layer 1
        x1 = self.conv1(x, codes)
        x2 = self.w1(x.view(mini_batch_size * n_env, -1, 1))
        x = x1 + x2.view(mini_batch_size, n_env, -1)

        # Projection with Q
        x = self.fc1(x.view(mini_batch_size * n_env, -1, 1))
        x = x.view(mini_batch_size, n_env, -1)
        x = self.a1(x)
        x = self.fc2(x.view(mini_batch_size * n_env, -1, 1))
        x = x.view(mini_batch_size, n_env, -1)
        x = x.view(mini_batch_size, -1)

        return x.squeeze(-1)

    def get_grid(self, mini_batch_size, n_env, size_x, device):
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x).repeat([mini_batch_size, n_env, 1])
        return gridx.to(device)


class GroupSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, code_channels, n_env, modes1, modes2):
        super().__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        Using learnable splitter.   
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.code_channels = code_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.n_env = n_env

        self.com_weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.com_weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.code_weights1 = nn.Parameter(self.scale * torch.rand(code_channels * in_channels, out_channels, self.modes1, self.modes2, 2))
        self.code_weights2 = nn.Parameter(self.scale * torch.rand(code_channels * in_channels, out_channels, self.modes1, self.modes2, 2))
        self.filter_weights1 = nn.Parameter(-3. * torch.ones(self.modes1, self.modes2, dtype=torch.float32), requires_grad=True)
        self.filter_weights2 = nn.Parameter(-3. * torch.ones(self.modes1, self.modes2, dtype=torch.float32), requires_grad=True)

    def compl_mul2d(self, inputs, com_weights, env_weights, lam):
        return (lam * torch.einsum("beixy,ioxy->beoxy", inputs, torch.view_as_complex(com_weights)) +
                (1. - lam) * torch.einsum("beixy,eioxy->beoxy", inputs, torch.view_as_complex(env_weights)))

    def forward(self, x, codes):
        batchsize = x.shape[0] // self.n_env
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        x_ft = x_ft.reshape(batchsize, self.n_env, self.in_channels, x.size(-2), x.size(-1) // 2 + 1)
        out_ft = torch.zeros(batchsize, self.n_env, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)

        filter_lam1 = F.hardsigmoid(self.filter_weights1)
        com_weights1 = self.com_weights1.reshape(self.in_channels, self.out_channels, self.modes1, self.modes2, 2)
        code_weights1 = self.code_weights1.reshape(self.code_channels, self.in_channels, self.out_channels, self.modes1, self.modes2, 2)
        env_weights1 = torch.einsum("ec,cioxyz->eioxyz", codes, code_weights1)
        out_ft[:, :, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :, :self.modes1, :self.modes2], com_weights1, env_weights1, filter_lam1)

        filter_lam2 = F.hardsigmoid(self.filter_weights1)
        com_weights2 = self.com_weights2.reshape(self.in_channels, self.out_channels, self.modes1, self.modes2, 2)
        code_weights2 = self.code_weights2.reshape(self.code_channels, self.in_channels, self.out_channels, self.modes1, self.modes2, 2)
        env_weights2 = torch.einsum("ec,cioxyz->eioxyz", codes, code_weights2)
        out_ft[:, :, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, :, -self.modes1:, :self.modes2], com_weights2, env_weights2, filter_lam2)

        out_ft = out_ft.reshape(batchsize * self.n_env, self.out_channels, x.size(-2), x.size(-1) // 2 + 1)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

# ---------------------------------------------- Gray --------------------------------------------------------------
class GrayGroupFNO2d(nn.Module):
    def __init__(self, input_dim, output_dim, code_dim, n_env, modes1=16, modes2=12, width=10, groups=1, nl='swish'): # modes1: 4
        super().__init__()
        self.width = width
        self.n_env = n_env
        self.groups = groups
        self.output_dim = output_dim

        self.fc0 = nn.Conv2d(in_channels=input_dim + 2, out_channels=self.width, kernel_size=1)
        self.conv0 = GroupSpectralConv2d(self.width, self.width, code_dim, n_env, modes1, modes2)
        self.conv1 = GroupSpectralConv2d(self.width, self.width, code_dim, n_env, modes1, modes2)
        self.conv2 = GroupSpectralConv2d(self.width, self.width, code_dim, n_env, modes1, modes2)
        self.conv3 = GroupSpectralConv2d(self.width, self.width, code_dim, n_env, modes1, modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1, groups=groups)
        self.w1 = nn.Conv2d(self.width, self.width, 1, groups=groups)
        self.w2 = nn.Conv2d(self.width, self.width, 1, groups=groups)
        self.w3 = nn.Conv2d(self.width, self.width, 1, groups=groups)
        self.a0 = GroupActivation(nl, groups=n_env)
        self.a1 = GroupActivation(nl, groups=n_env)
        self.a2 = GroupActivation(nl, groups=n_env)
        self.a3 = GroupActivation(nl, groups=n_env)
        self.fc1 = nn.Conv2d(self.width, (output_dim // 2) * self.width, kernel_size=1, groups=width)
        self.fc2 = nn.Conv2d((output_dim // 2) * self.width, output_dim, kernel_size=1)

    def forward(self, x, codes):
        x = x.view(x.size(0), self.n_env, self.output_dim, *x.shape[-2:])
        minibatch_size, n_env, state_dim, size_x, size_y = x.size()
        grid = self.get_grid(minibatch_size, n_env, size_x, size_y, x.device)
        x = torch.cat((x, grid), dim=2)
        x = x.view(minibatch_size * n_env, *x.shape[2:])

        # Lift with P
        x = self.fc0(x)
        # Fourier layer 0
        x1 = self.conv0(x, codes)
        x2 = self.w0(x)
        x = self.a0((x1 + x2).view(minibatch_size, n_env, *x1.shape[1:]))
        x = x.view(minibatch_size * n_env, *x.shape[2:])

        # Fourier layer 1
        x1 = self.conv1(x, codes)
        x2 = self.w1(x)
        x = self.a1((x1 + x2).view(minibatch_size, n_env, *x1.shape[1:]))
        x = x.view(minibatch_size * n_env, *x.shape[2:])

        # Fourier layer 2
        x1 = self.conv2(x, codes)
        x2 = self.w2(x)
        x = self.a2((x1 + x2).view(minibatch_size, n_env, *x1.shape[1:]))
        x = x.view(minibatch_size * n_env, *x.shape[2:])

        # Fourier layer 3
        x1 = self.conv3(x, codes)
        x2 = self.w3(x)
        x = x1 + x2

        # Projection with Q
        x = self.a3(self.fc1(x).view(minibatch_size, n_env, *x1.shape[1:]))
        x = self.fc2(x.view(minibatch_size * n_env, *x.shape[2:]))
        x = x.view(minibatch_size, n_env, *x.shape[1:])
        return x.view(minibatch_size, n_env * self.output_dim, size_x, size_y)

    def get_grid(self, batchsize, n_env, size_x, size_y, device):
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, n_env, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, n_env, size_x, 1])
        return torch.stack([gridx, gridy], dim=2).to(device)


# ---------------------------------------------- NS -----------------------------------------------------------------
class GroupFNO2d(nn.Module):
    def __init__(self, input_dim, output_dim, code_dim, n_env, modes1=16, modes2=12, width=10, groups=1, nl='swish'):
        """
        modes1: 4
        """
        super().__init__()
        self.width = width
        self.n_env = n_env
        self.groups = groups

        self.fc0 = nn.Conv2d(in_channels=input_dim + 2, out_channels=self.width, kernel_size=1)
        self.conv0 = GroupSpectralConv2d(self.width, self.width, code_dim, n_env, modes1, modes2)
        self.conv1 = GroupSpectralConv2d(self.width, self.width, code_dim, n_env, modes1, modes2)
        self.conv2 = GroupSpectralConv2d(self.width, self.width, code_dim, n_env, modes1, modes2)
        self.conv3 = GroupSpectralConv2d(self.width, self.width, code_dim, n_env, modes1, modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1, groups=groups)
        self.w1 = nn.Conv2d(self.width, self.width, 1, groups=groups)
        self.w2 = nn.Conv2d(self.width, self.width, 1, groups=groups)
        self.w3 = nn.Conv2d(self.width, self.width, 1, groups=groups)
        self.a0 = GroupActivation(nl, groups=n_env)
        self.a1 = GroupActivation(nl, groups=n_env)
        self.a2 = GroupActivation(nl, groups=n_env)
        self.a3 = GroupActivation(nl, groups=n_env)
        self.fc1 = nn.Conv2d(self.width, output_dim * self.width, kernel_size=1, groups=width)
        self.fc2 = nn.Conv2d(output_dim * self.width, output_dim, kernel_size=1)

    def forward(self, x, codes):
        minibatch_size, n_env, size_x, size_y = x.size()
        grid = self.get_grid(minibatch_size, n_env, size_x, size_y, x.device)
        x = torch.cat((x.unsqueeze(2), grid), dim=2)
        x = x.view(minibatch_size * n_env, *x.shape[2:])

        # Lift with P
        x = self.fc0(x)
        # Fourier layer 0
        x1 = self.conv0(x, codes)
        x2 = self.w0(x)
        x = self.a0((x1 + x2).view(minibatch_size, n_env, *x1.shape[1:]))
        x = x.view(minibatch_size * n_env, *x.shape[2:])

        # Fourier layer 1
        x1 = self.conv1(x, codes)
        x2 = self.w1(x)
        x = self.a1((x1 + x2).view(minibatch_size, n_env, *x1.shape[1:]))
        x = x.view(minibatch_size * n_env, *x.shape[2:])

        # Fourier layer 2
        x1 = self.conv2(x, codes)
        x2 = self.w2(x)
        x = self.a2((x1 + x2).view(minibatch_size, n_env, *x1.shape[1:]))
        x = x.view(minibatch_size * n_env, *x.shape[2:])

        # Fourier layer 3
        x1 = self.conv3(x, codes)
        x2 = self.w3(x)
        x = x1 + x2

        # Projection with Q
        x = self.a3(self.fc1(x).view(minibatch_size, n_env, *x1.shape[1:]))
        x = self.fc2(x.view(minibatch_size * n_env, *x.shape[2:]))
        return x.view(minibatch_size, n_env, size_x, size_y)

    def get_grid(self, batchsize, n_env, size_x, size_y, device):
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, n_env, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, n_env, size_x, 1])
        return torch.stack([gridx, gridy], dim=2).to(device)