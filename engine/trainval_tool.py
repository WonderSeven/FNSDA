import pdb
import time

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from engine.common import batch_transform
from engine.common import AverageMeter, ProgressMeter
from models.loss_func import *


def train(cfg, epoch, algorithm, train_loader, use_cuda=True, print_freq=50, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    total_losses = AverageMeter('Total Loss', ':.4e')
    original_losses = AverageMeter('Original Loss', ':.4e')
    regular_losses = AverageMeter('Regularization Loss', ':.4e')

    progress = ProgressMeter(len(train_loader), batch_time, total_losses, original_losses, regular_losses,
                             prefix="Epoch: [{}]".format(epoch))

    algorithm.train()
    end = time.time()
    for iteration, data in enumerate(train_loader, 0):
        state, t = data["state"], data["t"]
        if use_cuda:
            state, t = state.cuda(), t.cuda()

        batch_size = state.size(0)
        inputs = batch_transform(state, cfg[cfg.trainer.data_name]['train_batch_size'])
        pred, total_loss, original_loss, regular_loss = algorithm.update(inputs, t[0])
        total_losses.update(total_loss.item(), batch_size)
        original_losses.update(original_loss.item(), batch_size)
        regular_losses.update(regular_loss.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % print_freq == 0 and iteration != 0:
            progress.print(iteration)

    progress.print(epoch)

    return total_losses


def val(cfg, algorithm, val_loader, val_ood_loader, use_cuda=True, writer=None):
    losses_MSE = AverageMeter('MSE', ':.4e')
    losses_RMSE_mean = AverageMeter('RMSE Mean', ':.4e')
    losses_RMSE_var = AverageMeter('RMSE Var', ':.4e')
    losses_NMSE = AverageMeter('N-MSE', ':.4e')
    losses_MAPE_mean = AverageMeter('MAPE Mean', ':.4e')
    losses_MAPE_var = AverageMeter('MAPE Var', ':.4e')
    dataloader_test_list = [(val_loader, "ind"), (val_ood_loader, "ood")] if val_ood_loader else [(val_loader, "ind")]

    algorithm.eval()
    for (dataloader_test_instance, test_type) in dataloader_test_list:

        for iteration, data_test in enumerate(dataloader_test_instance, 0):
            with torch.no_grad():
                state, t = data_test["state"], data_test["t"]
                if use_cuda:
                    state, t = state.cuda(), t.cuda()

                targets = state
                batch_size = state.size(0)
                inputs = batch_transform(state, cfg[cfg.trainer.data_name]['eval_batch_size'])
                pred = algorithm.infer(inputs, t[0])
                loss_MSE = F.mse_loss(pred, targets)
                loss_NMSE = NMSE(pred, targets)
                loss_rmse_mean, loss_rmse_var = RMSE(pred, targets)
                loss_mape_mean, loss_mape_var = MAPE(pred, targets)

                losses_MSE.update(loss_MSE, batch_size)
                losses_RMSE_mean.update(loss_rmse_mean, batch_size)
                losses_RMSE_var.update(loss_rmse_var, batch_size)
                losses_NMSE.update(loss_NMSE, batch_size)
                losses_MAPE_mean.update(loss_mape_mean, batch_size)
                losses_MAPE_var.update(loss_mape_var, batch_size)

    return losses_MSE, (losses_RMSE_mean, losses_RMSE_var), losses_NMSE, (losses_MAPE_mean, losses_MAPE_var)


def adp_ood(cfg, epoch, algorithm, train_loader, use_cuda=True, print_freq=50, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    total_losses = AverageMeter('Total Loss', ':.4e')
    original_losses = AverageMeter('Original Loss', ':.4e')
    regular_losses = AverageMeter('Regularization Loss', ':.4e')

    progress = ProgressMeter(len(train_loader), batch_time, total_losses, original_losses, regular_losses,
                             prefix="Epoch: [{}]".format(epoch))

    minibatch_size = 1

    if cfg.trainer.task == 'extra':
        minibatch_size = cfg[cfg.trainer.data_name]['eval_batch_size']

    end = time.time()
    algorithm.train()
    for iteration, data in enumerate(train_loader, 0):
        state, t = data["state"], data["t"]
        if use_cuda:
            state, t = state.cuda(), t.cuda()
        if cfg.trainer.task == 'extra':
            if cfg.trainer.data_name in ['lotka', 'g_osci']:
                state = state[:, :, :10]
                t = t[:, :10]
            elif cfg.trainer.data_name in ['gray', 'navier']:
                state = state[:, :, :2]
                t = t[:, :2]

        inputs = batch_transform(state, minibatch_size)

        pred, total_loss, original_loss, regular_loss = algorithm.adapt(inputs, t[0])

        total_losses.update(total_loss.item(), minibatch_size)
        original_losses.update(original_loss.item(), minibatch_size)
        regular_losses.update(regular_loss.item(), minibatch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % print_freq == 0 and iteration != 0:
            progress.print(iteration)

    return total_losses


def val_ood(cfg, algorithm, val_ood_loader, use_cuda=True, writer=None):
    losses_MSE = AverageMeter('MSE', ':.4e')
    losses_RMSE_mean = AverageMeter('RMSE Mean', ':.4e')
    losses_RMSE_var = AverageMeter('RMSE Var', ':.4e')
    losses_NMSE = AverageMeter('N-MSE', ':.4e')
    losses_MAPE_mean = AverageMeter('MAPE Mean', ':.4e')
    losses_MAPE_var = AverageMeter('MAPE Var', ':.4e')
    loss_test_env = torch.zeros(cfg.trainer.n_env)
    loss_relative_env = torch.zeros(cfg.trainer.n_env)

    minibatch_size = cfg[cfg.trainer.data_name]['eval_batch_size']


    algorithm.eval()
    for iteration, data_test in enumerate(val_ood_loader, 0):
        with torch.no_grad():
            state, t = data_test["state"], data_test["t"]
            if use_cuda:
                state, t = state.cuda(), t.cuda()

            if cfg.trainer.task == 'extra':
                if cfg.trainer.data_name in ['lotka', 'g_osci']:
                    state = state[:, :, 10:]
                    t = t[:, 10:]
                elif cfg.trainer.data_name in ['gray', 'navier']:
                    state = state[:, :, 2:]
                    t = t[:, 2:]

            targets = state
            inputs = batch_transform(state, minibatch_size)
            pred = algorithm.infer(inputs, t[0])
            # record
            loss_MSE = F.mse_loss(pred, targets)
            loss_NMSE = NMSE(pred, targets)
            loss_rmse_mean, loss_rmse_var = RMSE(pred, targets)
            loss_mape_mean, loss_mape_var = MAPE(pred, targets)
            loss_Env_MSE, loss_Env_NMSE = Env_NMSE(pred, targets, minibatch_size)
            # Update log
            losses_MSE.update(loss_MSE, minibatch_size)
            losses_RMSE_mean.update(loss_rmse_mean, minibatch_size)
            losses_RMSE_var.update(loss_rmse_var, minibatch_size)
            losses_NMSE.update(loss_NMSE, minibatch_size)
            losses_MAPE_mean.update(loss_mape_mean, minibatch_size)
            losses_MAPE_var.update(loss_mape_var, minibatch_size)
            loss_test_env += loss_Env_MSE
            loss_relative_env += loss_Env_NMSE

    loss_test_env /= iteration + 1
    loss_test_env /= iteration + 1

    return losses_MSE, (losses_RMSE_mean, losses_RMSE_var), losses_NMSE, (losses_MAPE_mean, losses_MAPE_var), loss_test_env, loss_relative_env
