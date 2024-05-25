import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from engine.common import batch_transform_loss


class WeightInterpolationLoss(nn.Module):
    def __init__(self, model_func):
        super(WeightInterpolationLoss, self).__init__()
        self._make_params()
    def _make_params(self):
        self.u, self.v = dict(), dict()
        for i, module in enumerate(self.module):
            for name, w in module.named_parameters():
                if name.find('bias') == -1 and name.find('beta') == -1:
                    height = w.data.shape[0]
                    width = w.view(height, -1).data.shape[1]

                    u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False).to(self.device)
                    v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False).to(self.device)

                    self.u[f'{i},{name}'] = u
                    self.v[f'{i},{name}'] = v

def NMSE(pred, targets):
    # Normalized mean squared error
    raw_loss_NL2 = torch.square(pred - targets) / torch.square(targets)
    loss_NL2 = torch.nanmean(raw_loss_NL2)
    return loss_NL2

def MAPE(pred, targets):
    # Normalized mean l1 error
    raw_loss_NL1 = torch.abs(pred - targets) / torch.abs(targets)
    loss_NL1_mean = torch.nanmean(raw_loss_NL1)
    num_examples = pred.size(0)
    diff = torch.nanmean(raw_loss_NL1.reshape(num_examples, -1), dim=-1)
    loss_NL1_std = diff[~(torch.isnan(diff))].std()
    return loss_NL1_mean, loss_NL1_std

def RMSE(pred, targets):
    rmse_loss = torch.sqrt(torch.mean(F.mse_loss(pred, targets, reduction='none'), dim=tuple(range(pred.ndimension())[1:])))
    rmse_loss_mean, rmse_loss_std = rmse_loss.mean(), rmse_loss.std()
    return rmse_loss_mean, rmse_loss_std


def Env_NMSE(pred, targets, batch_size):
    raw_loss_relative = torch.abs(pred - targets) / torch.abs(targets)
    # reformat into [batch_size, n_env, x_dim, time_steps]
    pred = batch_transform_loss(pred, batch_size)
    targets = batch_transform_loss(targets, batch_size)
    raw_loss_relative = batch_transform_loss(raw_loss_relative, batch_size)

    dim = list(range(pred.dim()))
    dim.remove(1)
    loss_test_env = F.mse_loss(pred, targets, reduction='none').mean(dim=dim).cpu()
    loss_relative_env = raw_loss_relative.nanmean(dim=dim).cpu()
    return loss_test_env, loss_relative_env
