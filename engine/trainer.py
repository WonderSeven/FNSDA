# -*- coding:utf-8 -*-
import os
import pdb
import json
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

import engine.inits as inits
from engine.common import count_parameters
from engine import trainval_tool as tv


class Trainer(object):
    def __init__(self, cfg):
        print('Constructing components...')
        # basic settings
        self.cfg = cfg
        self.gpu_ids = str(cfg.gpu_ids).lstrip('(').rstrip(')') if isinstance(cfg.gpu_ids, tuple) else str(cfg.gpu_ids)
        self.task = cfg.trainer.task
        self.epochs = cfg.trainer.train_epochs
        self.stage = cfg.trainer.mode
        self.output_dir = inits.get_record_dir(cfg, name=None)

        # seed and stage
        inits.set_seed(cfg.trainer.seed)

        # To cuda
        print('GPUs id:{}'.format(self.gpu_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_ids
        self.use_cuda = torch.cuda.is_available()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.train_loader, self.test_loader, n_env = inits.get_dataloaders(cfg, self.stage)
        self.cfg['trainer']['n_env'] = cfg['trainer']['n_env'] = n_env
        self.algorithm = inits.get_algorithm(cfg)

        # multi gpu
        if len(self.gpu_ids.split(',')) > 1:
            self.algorithm = nn.DataParallel(self.algorithm)
            print('GPUs:', torch.cuda.device_count())
            print('Using CUDA...')
        if self.use_cuda:
            self.algorithm.cuda()

        self.start_epoch = 0
        self.best_val = np.inf
        self.print_freq = cfg.trainer.print_freq
        self.val_freq = cfg.trainer.val_freq
        self.test_epoch = cfg.trainer.test_epoch

        # checkpoint
        self.checkpointer = inits.get_checkpointer(cfg, self.algorithm, self.stage, self.output_dir)
        self.set_training_stage(self.stage)

        # log
        self.writer = None
        self.logger = inits.get_logger(cfg, self.output_dir)
        self.logger.setLevel(logging.INFO)
        self.logger.info('{} - {} - {}'.format(self.cfg.trainer.data_name, self.task, self.cfg.algorithm.name))
        self.logger.info(f"net parameters: {count_parameters(self.algorithm.vector_field)} => {dict(self.algorithm.vector_field.named_parameters()).keys()}")

    def set_training_stage(self, stage):
        stage = stage.strip().lower()
        if stage == 'train':
            self.stage = 2

        elif stage == 'val' or stage == 'test':
            self.stage = 1
            self.epochs = self.cfg.trainer.adapt_epochs if self.cfg.trainer.adaptation else 0
            self.checkpointer.load_model(self._get_load_name(self.test_epoch))
            if self.cfg.algorithm.name.lower() in ['coda', 'leads', 'gi', 'gi_adp', 'fourier']:
                self.algorithm.set_no_grad()

        elif stage == 'continue':
            self.stage = 2
            start_model = self._get_load_name(-2)
            self.start_epoch = self.checkpointer.load_model(start_model)

    @staticmethod
    def _get_load_name(epoch=-1):
        if epoch == -1:
            model_name = 'best'
        elif epoch == -2:
            model_name = 'last'
        elif epoch == -3:
            model_name = 'inter'
        elif epoch == -4:
            model_name = 'extra'
        else:
            model_name = str(epoch)
        return model_name

    def _train_net(self, epoch):
        losses = tv.train(self.cfg, epoch, self.algorithm, self.train_loader, use_cuda=self.use_cuda,
                          print_freq=self.print_freq, writer=self.writer)
        return losses

    def _val_net(self, dataloader):
        return tv.val(self.cfg, self.algorithm, dataloader, val_ood_loader=None, use_cuda=self.use_cuda, writer=self.writer)

    def _adp_ood_net(self, epoch, dataloader):
        losses = tv.adp_ood(self.cfg, epoch, self.algorithm, dataloader, use_cuda=self.use_cuda,
                                print_freq=self.print_freq, writer=self.writer)
        return losses

    def _val_ood_net(self, dataloader):
        return tv.val_ood(self.cfg, self.algorithm, dataloader, use_cuda=self.use_cuda, writer=self.writer)


    def train(self):
        if self.stage >= 2:
            for epoch_item in range(self.start_epoch, self.epochs):
                train_loss = self._train_net(epoch_item)

                if (epoch_item % self.val_freq == 0) and (epoch_item!=0):
                    val_MSE, val_RMSE, val_NMSE, val_NL1 = self._val_net(self.test_loader)
                    if val_MSE.avg.item() < self.best_val:
                        self.best_val = val_MSE.avg.item()

                        if self.checkpointer is not None:
                            self.checkpointer.save_model('best', epoch_item)

                    self.logger.info('==================================== Epoch %d ====================================' % epoch_item)
                    self.logger.info('Epoch:{} || Train: {}'.format(epoch_item, train_loss))
                    self.logger.info('Epoch:{} || Val MSE: {:.4E}, RMSE: {:.4E} ± {:.4E}, NMSE: {:.4f}, N-L1: {:.4E} ± {:.4E} || Best Acc@1:{:6.3f}'.format(
                        epoch_item, val_MSE.avg.item(), val_RMSE[0].avg.item(), val_RMSE[1].avg.item(),
                        val_NMSE.avg.item(), val_NL1[0].avg.item(), val_NL1[1].avg.item(), self.best_val))

            if self.checkpointer is not None:
                self.checkpointer.save_model('last', self.epochs)

        else:
            self.logger.info('================================ OOD Generalization ================================')
            if self.cfg.trainer.adaptation:
                last_train_loss = float('inf')
                done = False

                for epoch_item in range(self.epochs):
                    train_loss = self._adp_ood_net(epoch_item, self.test_loader if self.task == 'extra' else self.train_loader)

                    difference = abs(train_loss.avg - last_train_loss)
                    if difference < 1e-12:
                        done = True
                    last_train_loss = train_loss.avg

                    if (epoch_item % self.val_freq == 0) and (epoch_item != 0):
                        val_MSE, val_RMSE, val_NMSE, val_NL1, val_env_loss, val_env_loss_relative = self._val_ood_net(self.test_loader)
                        self.logger.info('==================================== Epoch %d ====================================' % epoch_item)
                        self.logger.info('Epoch:{} || Train: {}'.format(epoch_item, train_loss))
                        self.logger.info('Epoch:{} || Val MSE: {:.4E}, RMSE: {:.4E} ± {:.4E}, NMSE: {:.4f}, N-L1: {:.4E} ± {:.4E} || Val Env : {}, Val Env Relative: {}'.format(
                            epoch_item, val_MSE.avg.item(), val_RMSE[0].avg.item(), val_RMSE[1].avg.item(),
                            val_NMSE.avg.item(), val_NL1[0].avg.item(), val_NL1[1].avg.item(), val_env_loss, val_env_loss_relative))

                        self.checkpointer.save_model(self.cfg.trainer.task, epoch_item)

            self.logger.info('==================================== Final Test ====================================')
            val_MSE, val_RMSE, val_NMSE, val_NL1, _, _ = self._val_ood_net(self.test_loader)
            self.logger.info('Final || Val MSE: {:.4E}, RMSE: {:.4E} ± {:.4E}, NMSE: {:.4f}, N-L1: {:.4E} ± {:.4E}'.format(
                val_MSE.avg.item(), val_RMSE[0].avg.item(), val_RMSE[1].avg.item(), val_NMSE.avg.item(),
                val_NL1[0].avg.item(), val_NL1[1].avg.item()))
