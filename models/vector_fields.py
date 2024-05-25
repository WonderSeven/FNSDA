import pdb
import copy
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodules import GroupActivation, GroupSpectralConv1d, GroupSpectralConv2d
from engine.common import batch_transform, batch_transform_inverse


# =================================== Convolutional network ===================================
class GroupConvMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, groups=1, factor=1.0, nl="swish"):
        super().__init__()
        self.factor = factor
        self.net = nn.Sequential(
            nn.Conv1d(input_dim * groups, hidden_dim * groups, 1, groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv1d(hidden_dim * groups, hidden_dim * groups, 1, groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv1d(hidden_dim * groups, hidden_dim * groups, 1, groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv1d(hidden_dim * groups, output_dim * groups, 1, groups=groups),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        return self.net(x).squeeze(-1) * self.factor


class GroupConv(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, groups=1, factor=1.0, nl="swish", size=64, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.output_dim = output_dim
        self.factor = factor
        self.hidden_dim = hidden_dim
        self.size = size
        self.net = nn.Sequential(
            nn.Conv2d(input_dim * groups, hidden_dim * groups, kernel_size=kernel_size, padding=padding, padding_mode='circular', groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv2d(hidden_dim * groups, hidden_dim * groups, kernel_size=kernel_size, padding=padding, padding_mode='circular', groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv2d(hidden_dim * groups, hidden_dim * groups, kernel_size=kernel_size, padding=padding, padding_mode='circular', groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv2d(hidden_dim * groups, output_dim * groups, kernel_size=kernel_size, padding=padding, padding_mode='circular', groups=groups)
        )

    def forward(self, x):
        x = self.net(x)
        # print(x.abs().mean())
        x = x * self.factor
        return x


# =================================== Fourier network ===================================
class GroupFNO1d(nn.Module):
    def __init__(self, input_dim, output_dim, modes1=5, width=10, groups=1, nl='swish'):
        super().__init__()
        self.state_dim = input_dim
        self.width = width
        self.groups = groups

        self.p = nn.Conv1d(2 * input_dim * self.groups, self.width * self.groups, 1, groups=groups)
        self.conv0 = GroupSpectralConv1d(1, 1, modes1, groups)
        self.conv1 = GroupSpectralConv1d(1, 1, modes1, groups)
        self.conv2 = GroupSpectralConv1d(1, 1, modes1, groups)
        self.conv3 = GroupSpectralConv1d(1, 1, modes1, groups)
        self.w0 = nn.Conv1d(self.width * self.groups, self.width * self.groups, 1, groups=groups)
        self.w1 = nn.Conv1d(self.width * self.groups, self.width * self.groups, 1, groups=groups)
        self.w2 = nn.Conv1d(self.width * self.groups, self.width * self.groups, 1, groups=groups)
        self.w3 = nn.Conv1d(self.width * self.groups, self.width * self.groups, 1, groups=groups)
        self.a0 = GroupActivation(nl, groups=groups)
        self.a1 = GroupActivation(nl, groups=groups)
        self.a2 = GroupActivation(nl, groups=groups)
        self.a3 = GroupActivation(nl, groups=groups)
        self.fc1 = nn.Conv1d(self.width * self.groups, self.width * self.groups, 1, groups=groups)
        self.fc2 = nn.Conv1d(self.width * self.groups, output_dim * self.groups, 1, groups=groups)

    def forward(self, x):
        """
        :param x: [batch_size, n_env*state_dim]
        :return:
        """
        x = x.view(x.size(0), -1, self.state_dim)
        mini_batch_size, n_env, size_x = x.size()
        grid = self.get_grid(mini_batch_size, n_env, size_x, x.device)
        x = torch.cat([x, grid], dim=-1)
        x = x.view(mini_batch_size * n_env, -1, 1)

        # Lift with P
        x = self.p(x)
        x = x.view(mini_batch_size, n_env, -1)
        # Fourier layer 0
        x1 = self.conv0(x)
        x2 = self.w0(x.view(mini_batch_size * n_env, -1, 1))
        x = self.a0(x1 + x2.view(mini_batch_size, n_env, -1))
        # Fourier layer 1
        x1 = self.conv1(x)  # [4, 9, 20]
        x2 = self.w1(x.view(mini_batch_size * n_env, -1, 1))
        x = self.a1(x1 + x2.view(mini_batch_size, n_env, -1))
        # Fourier layer 2
        x1 = self.conv2(x)
        x2 = self.w2(x.view(mini_batch_size * n_env, -1, 1))
        x = self.a2(x1 + x2.view(mini_batch_size, n_env, -1))

        # Fourier layer 3
        x1 = self.conv3(x)
        x2 = self.w3(x.view(mini_batch_size * n_env, -1, 1))
        x = x1 + x2.view(mini_batch_size, n_env, -1)

        # Projection with Q
        x = self.fc1(x.view(mini_batch_size * n_env, -1, 1))
        x = x.view(mini_batch_size, n_env, -1)
        x = self.a3(x)
        x = self.fc2(x.view(mini_batch_size * n_env, -1, 1))
        x = x.view(mini_batch_size, n_env, -1)
        x = x.view(mini_batch_size, -1)

        return x.squeeze(-1)

    def get_grid(self, mini_batch_size, n_env, size_x, device):
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x).repeat([mini_batch_size, n_env, 1])
        return gridx.to(device)


# 2D ------------------------------------------------------------------------------------------------------------------
class GroupFNO2d(nn.Module):
    def __init__(self, input_dim, output_dim, modes1=12, modes2=12, width=10, groups=1, nl='swish'):
        super().__init__()
        self.width = width
        self.groups = groups
        self.fc0 = nn.Conv2d((input_dim + 2) * self.groups, self.width * self.groups, 1, groups=groups)
        self.conv0 = GroupSpectralConv2d(self.width, self.width, modes1, modes2, groups)
        self.conv1 = GroupSpectralConv2d(self.width, self.width, modes1, modes2, groups)
        self.conv2 = GroupSpectralConv2d(self.width, self.width, modes1, modes2, groups)
        self.conv3 = GroupSpectralConv2d(self.width, self.width, modes1, modes2, groups)
        self.w0 = nn.Conv2d(self.width * self.groups, self.width * self.groups, 1, groups=groups)
        self.w1 = nn.Conv2d(self.width * self.groups, self.width * self.groups, 1, groups=groups)
        self.w2 = nn.Conv2d(self.width * self.groups, self.width * self.groups, 1, groups=groups)
        self.w3 = nn.Conv2d(self.width * self.groups, self.width * self.groups, 1, groups=groups)
        self.a0 = GroupActivation(nl, groups=groups)
        self.a1 = GroupActivation(nl, groups=groups)
        self.a2 = GroupActivation(nl, groups=groups)
        self.a3 = GroupActivation(nl, groups=groups)
        self.fc1 = nn.Conv2d(self.width * self.groups, 128 * self.groups, 1, groups=groups)
        self.fc2 = nn.Conv2d(128 * self.groups, output_dim * self.groups, 1, groups=groups)

    def forward(self, x):
        """
        @ params: x  batchsize x n_env * c x h x w
        NS: [250, 1, 32, 32]
        """
        minibatch_size = x.shape[0]
        x = batch_transform_inverse(x, self.groups)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[-2], x.shape[-1]
        grid = self.get_grid(batchsize, size_x, size_y, x.device)
        x = torch.cat((x, grid), dim=1)
        x = batch_transform(x, minibatch_size)

        # Lift with P
        x = self.fc0(x)
        # Fourier layer 0
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = self.a0(x1 + x2)
        # Fourier layer 1
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = self.a1(x1 + x2)
        # Fourier layer 2
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = self.a2(x1 + x2)
        # Fourier layer 3
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # Projection with Q
        x = self.a3(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_grid(self, batchsize, size_x, size_y, device):
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)
