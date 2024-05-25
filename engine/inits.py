import os
import pdb

import math
import copy
import random
import numpy as np
from itertools import product

import torch
import torch.nn as nn

from datasets import DataLoaderODE
from engine.common import Checkpointer
from engine.common import create_logger, add_filehandler, format_time
from engine import configs
import models
import datasets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_function(registry, name):
    if name in registry:
        return registry[name]
    else:
        raise Exception("{} does not support [{}], valid keys : {}".format(registry, name, list(registry.keys())))

def get_record_dir(cfg, name=None):
    output_dir = cfg['trainer']['output_dir']

    if name is not None:
        output_dir = os.path.join(output_dir, name)

    if cfg.trainer.record:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    return output_dir

def get_algorithm(cfg):
    name = cfg.algorithm.name.lower()
    func = get_function(configs.Algorithms, name)

    method_name_dict = {
        'RK45': 'rk4',
        'euler': 'euler'
    }

    hparams = dict(cfg.solver.params)
    hparams.update(cfg.scheduler.params)

    hparams.update({
        'opt_name'      : cfg.solver.optimizer.lower(),
        'lr'            : float(cfg.solver.base_lr) if isinstance(cfg.solver.base_lr, str) else cfg.solver.base_lr,
        'weight_decay'  : float(cfg.solver.weight_decay) if isinstance(cfg.solver.weight_decay, str) else cfg.solver.weight_decay,
        'criterion'     : cfg.solver.criterion,
        'scheduler_name': cfg.scheduler.name.lower()
    })

    if cfg.scheduler.name.lower() == 'cos' and cfg.trainer.mode.lower() in ['val', 'test']:
        # During model adaptation, warmup=0
        hparams.update({'epochs': cfg.trainer.adapt_epochs, 'warmup': 0})

    is_ode = cfg.trainer.data_name.lower() in ["lotka", "g_osci"]

    args = dict(cfg.algorithm.params)

    args.update({
        'data_name' : cfg.trainer.data_name,
        'is_ode'    : is_ode,
        'state_dim' : cfg[cfg.trainer.data_name]['state_dim'],
        'hidden_dim': cfg.algorithm.hidden_dim,
        'n_env'     : cfg.trainer.n_env,
        'method'    : cfg.solver.method,
        'factor'    : float(cfg[cfg.trainer.data_name]['factor']),
        'nl'        : cfg.algorithm.nl,
        'size'      : 0 if is_ode else cfg[cfg.trainer.data_name]['size'],
        'hparams'   : hparams
    })

    if name in ['erm', 'coda', 'leads', 'foca', 'gi_adp', 'fourier', 'fourier_rand']:
        args.update({
            'method': method_name_dict[cfg[cfg.trainer.data_name]['method']],
        })

    if name == 'coda':
        args.update({
            'is_layer'  : (args['layers'][0] != -1),
        })

    return func(**args)

def get_loss_func(cfg):
    name = cfg.solver.criterion.lower()
    if name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif name == 'mse':
        return nn.MSELoss()
    elif name == 'l1':
        return nn.L1Loss()
    else:
        raise ValueError('criterion should be cross_entropy')

def get_datasets(cfg):
    data_name = cfg.trainer.data_name.lower()
    func = get_function(configs.Datasets, data_name)

    if data_name == 'lotka':
        data_params = cfg.lotka

        dataset_train_params = {
            "n_data_per_env": data_params.train_batch_size, "t_horizon": data_params.t_horizon, "dt": data_params.dt,
            "method": data_params.method, "split": 'train',
            "params": [
                {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.5},
                {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.5},
                {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.5},
                {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.75},
                {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 1.0},
                {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.75},
                {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 1.0},
                {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.75},
                {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 1.0}]}
        dataset_test_params = copy.deepcopy(dataset_train_params)
        dataset_test_params.update({"n_data_per_env": data_params.eval_batch_size,
                                    "split": "test"})
    elif data_name == 'g_osci':
        data_params = cfg.g_osci
        if data_params.k1_range_idx == 0:
            k1_range = [100, 90, 80]
        elif data_params.k1_range_idx == 1:
            k1_range = [100, 97.5, 95]
        elif data_params.k1_range_idx == 2:
            k1_range = [100, 95, 90]
        elif data_params.k1_range_idx == 3:
            k1_range = [100, 99.5, 99]
        K1_range = [1, 0.75, 0.5]

        dataset_train_params = {
            "n_data_per_env": data_params.train_batch_size, "t_horizon": data_params.t_horizon, "dt": data_params.dt,
            "method": data_params.method, "split": 'train',
            'params': [
                {'J0': 2.5, 'k1': k1, 'k2': 6, 'k3': 16, 'k4': 100, 'k5': 1.28, 'k6': 12, 'K1': K1, 'q': 4, 'N': 1,
                 'A': 4, 'kappa': 13, 'psi': 0.1, 'k': 1.8} for k1 in k1_range for K1 in K1_range]}
        dataset_test_params = copy.deepcopy(dataset_train_params)
        dataset_test_params.update({"n_data_per_env": data_params.eval_batch_size,
                                    "split": "test"})
    elif data_name == 'gray':
        data_params = cfg.gray
        record_file_path = './logs/baseline/gray/ERM'
        # record_file_path = cfg.trainer.output_dir

        dataset_train_params = {
            "n_data_per_env": data_params.train_batch_size, "t_horizon": data_params.t_horizon, "dt": data_params.dt,
            "method": data_params.method, "size": 32, "n_block": 3, "dx": 1, "split": "train",
            "buffer_file": f"{record_file_path}/gray_buffer_train.shelve",
            "params": [
                {"f": 0.03, "k": 0.062, "r_u": 0.2097, "r_v": 0.105},
                {"f": 0.039, "k": 0.058, "r_u": 0.2097, "r_v": 0.105},
                {"f": 0.03, "k": 0.058, "r_u": 0.2097, "r_v": 0.105},
                {"f": 0.039, "k": 0.062, "r_u": 0.2097, "r_v": 0.105}
            ]
        }
        dataset_test_params = copy.deepcopy(dataset_train_params)
        dataset_test_params.update({"n_data_per_env": data_params.eval_batch_size,
                                    "buffer_file": f"{record_file_path}/gray_buffer_test.shelve",
                                    "split": "test"})
    elif data_name == 'navier':
        data_params = cfg.navier
        record_file_path = './logs/baseline/navier/ERM'
        # record_file_path = cfg.trainer.output_dir

        tt = torch.linspace(0, 1, data_params.size + 1)[0:-1]
        X, Y = torch.meshgrid(tt, tt)
        dataset_train_params = {
            "n_data_per_env": data_params.train_batch_size, "t_horizon": data_params.t_horizon, "dt_eval": data_params.dt,
            "method": data_params.method, "size": data_params.size, "split": "train",
            "buffer_file": f"{record_file_path}/ns_buffer_train_3env_08-12_{data_params.size}.shelve",
            "params": [
                {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 8e-4},
                {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 9e-4},
                {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 1.0e-3},
                {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 1.1e-3},
                {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 1.2e-3},
            ]
        }

        dataset_test_params = copy.deepcopy(dataset_train_params)
        dataset_test_params.update({"n_data_per_env": data_params.eval_batch_size,
                                    "buffer_file": f"{record_file_path}/ns_buffer_test_3env_08-12_{data_params.size}.shelve",
                                    "split": "test"})
    else:
        raise ValueError('Not support dataset:{}'.format(cfg.trainer.data_name))

    dataset_train, dataset_test = func(**dataset_train_params), func(**dataset_test_params)

    return dataset_train, dataset_test

def get_ood_datasets(cfg):
    data_name = cfg.trainer.data_name.lower()
    func = get_function(configs.Datasets, data_name)

    if data_name == "lotka":
        data_params = cfg.lotka

        beta = [0.625, 0.625, 1.125, 1.125]
        delta = [0.625, 1.125, 0.625, 1.125]
        dataset_train_params = {"n_data_per_env": 1, "t_horizon": data_params.t_horizon, "dt": data_params.dt,
                                "method": data_params.method, "split": "train",
                                "params": [{"alpha": 0.5, "beta": beta_i, "gamma": 0.5, "delta": delta_i} for
                                           beta_i, delta_i in zip(beta, delta)]}
        dataset_test_params = copy.deepcopy(dataset_train_params)
        dataset_test_params.update({"n_data_per_env": data_params.eval_batch_size,
                                    "split": "test"})
    elif data_name == "g_osci":
        data_params = cfg.g_osci
        k1 = [85, 95]
        K1 = [0.625, 0.875]
        dataset_train_params = {'n_data_per_env': 1, 't_horizon': data_params.t_horizon, "dt": data_params.dt,
                                'method': data_params.method, 'split': 'train',
                                'params': [{'J0': 2.5, 'k1': k1_i, 'k2': 6, 'k3': 16, 'k4': 100, 'k5': 1.28, 'k6': 12,
                                            'K1': K1_i,
                                            'q': 4, 'N': 1, 'A': 4, 'kappa': 13, 'psi': 0.1, 'k': 1.8} for k1_i, K1_i in
                                           product(k1, K1)]}
        dataset_test_params = copy.deepcopy(dataset_train_params)
        dataset_test_params.update({"n_data_per_env": data_params.eval_batch_size,
                                    "split": "test"})
    elif data_name == "gray":
        data_params = cfg.gray
        record_file_path = './logs/baseline/gray/ERM'

        f = [0.033, 0.036]
        k = [0.059, 0.061]
        dataset_train_params = {"n_data_per_env": 1, "t_horizon": data_params.t_horizon, "dt": data_params.dt,
                                "size": data_params.size, "n_block": 3, "dx": 1,
                                "method": data_params.method,
                                "buffer_file": f"{record_file_path}/gray_buffer_train_ada.shelve",
                                "split": "train",
                                "params": [{"f": f_i, "k": k_i, "r_u": 0.2097, "r_v": 0.105} for f_i, k_i in
                                           product(f, k)]}
        dataset_test_params = copy.deepcopy(dataset_train_params)
        dataset_test_params.update({"n_data_per_env": data_params.eval_batch_size,
                                    "buffer_file": f"{record_file_path}/gray_buffer_test_ada.shelve",
                                    "split": "test"})
    elif data_name == "navier":
        data_params = cfg.navier
        record_file_path = './logs/baseline/navier/ERM'
        # record_file_path = cfg.trainer.output_dir

        tt = torch.linspace(0, 1, data_params.size + 1)[0:-1]
        X, Y = torch.meshgrid(tt, tt)
        f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

        viscs = [8.5e-4, 9.5e-4, 1.05e-3, 1.15e-3]

        dataset_train_params = {"n_data_per_env": 1, "t_horizon": data_params.t_horizon, "dt_eval": data_params.dt,
                                "size": data_params.size, "method": data_params.method,
                                "buffer_file": f"{record_file_path}/ns_buffer_ref_train_ada_{data_params.size}.shelve",
                                "split": "train", "params": [{"f": f, "visc": visc} for visc in viscs]}
        dataset_test_params = copy.deepcopy(dataset_train_params)
        dataset_test_params.update({"n_data_per_env": data_params.eval_batch_size,
                                    "buffer_file": f"{record_file_path}/ns_buffer_ref_test_ada{data_params.size}.shelve",
                                    "split": "test"})
    else:
        raise Exception(f"{data_name} does not exist")

    dataset_train, dataset_test = func(**dataset_train_params), func(**dataset_test_params)

    return dataset_train, dataset_test

def get_dataloaders(cfg, stage):
    if stage == 'train':
        dataset_train, dataset_test = get_datasets(cfg)
    elif stage in ['val', 'test']:
        dataset_train, dataset_test = get_ood_datasets(cfg)
    else:
        raise ValueError
    dataloader_train = DataLoaderODE(dataset_train, dataset_train.n_data_per_env, dataset_train.num_env)
    dataloader_test = DataLoaderODE(dataset_test, dataset_test.n_data_per_env, dataset_test.num_env, is_train=False)
    assert dataset_train.num_env == dataset_test.num_env

    return dataloader_train, dataloader_test, dataset_train.num_env

def get_checkpointer(cfg, algorithm, stage, output_path=None):
    if not cfg.trainer.record and stage == 'train':
        return None
    return Checkpointer(output_path, algorithm, cfg.trainer.seed)

def get_logger(cfg, output_path=None, name=None):
    logger = create_logger('DynamicPhysics')
    cur_time = format_time()
    if name is None:
        log_name = '{}_{}_{}_{}_seed{}_{}.txt'.format(cfg.trainer.mode, cfg.trainer.data_name, cfg.trainer.task,
                                                      cfg.algorithm.name, cfg.trainer.seed, cur_time)
    else:
        log_name = '{}_{}_{}_{}_{}_seed{}_{}.txt'.format(name, cfg.trainer.mode, cfg.trainer.data_name, cfg.trainer.task,
                                                      cfg.algorithm.name, cfg.trainer.seed, cur_time)

    log_path = os.path.join(output_path, log_name)

    if cfg.trainer.record:
        if os.path.exists(log_path):
            os.remove(log_path)
        # save config
        add_filehandler(logger, log_path)

    return logger

