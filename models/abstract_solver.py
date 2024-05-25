import abc
import pdb

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from models.schedulers import CosineWithWarmupScheduler


class AbstractSolver(nn.Module, abc.ABC):
    def __init__(self, state_dim, hidden_dim, data_name, n_env, hparams):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.data_name = data_name
        self.n_env = n_env
        self.hparams = hparams

    @abc.abstractmethod
    def _build(self):
        pass

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def infer(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def adapt(self, *args, **kwargs):
        pass

    def _get_optimizer(self, model_func):
        opt_name = self.hparams['opt_name']

        if opt_name == 'adam':
            return torch.optim.Adam([{'params': model_func.parameters()}], lr=self.hparams['lr'],
                                    weight_decay=self.hparams['weight_decay'])
        elif opt_name == 'sgd':
            return torch.optim.SGD([{'params': model_func.parameters()}], lr=self.hparams['lr'],
                                   weight_decay=self.hparams['weight_decay'], momentum=self.hparams['momentum'],
                                   nesterov=True)
        else:
            raise Exception("Not support opt : {}".format(opt_name))

    def _get_scheduler(self, optimizer):
        scheduler_name = self.hparams['scheduler_name']

        if scheduler_name == 'step':
            return lr_scheduler.StepLR(optimizer, step_size=self.hparams['step_size'], gamma=self.hparams['gamma'])
        elif scheduler_name == 'exp':
            return lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams['gamma'])
        elif scheduler_name == 'cos':
            return CosineWithWarmupScheduler(optimizer, num_warmup_steps=self.hparams['warmup'], num_training_steps=self.hparams['epochs']) # 500
        else:
            raise NotImplementedError('Scheduler name:{} is not supported!'.format(scheduler_name))

    def _get_loss_func(self):
        name = self.hparams['criterion']
        if name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif name == 'mse':
            return nn.MSELoss()
        elif name == 'l1':
            return nn.L1Loss()
        else:
            raise ValueError('criterion should in [cross_entropy, mse]')
