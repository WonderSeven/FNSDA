import pdb
import time
import torch

def format_time():
    return time.strftime("%Y-%m-%d-%H:%M", time.localtime(time.time()))


def batch_transform(batch, minibatch_size):
    # batch: b x c x t
    t = batch.shape[2:]
    new_batch = []
    for i in range(minibatch_size):
        sample = batch[i::minibatch_size]  # n_env x c x t
        sample = sample.reshape(-1, *t)
        new_batch.append(sample)
    return torch.stack(new_batch)  # minibatch_size x n_env * c x t

def batch_transform_loss(batch, minibatch_size):
    # batch: b x c x t
    t = batch.shape[2:]
    new_batch = []
    for i in range(minibatch_size):
        sample = batch[i::minibatch_size]  # n_env x c x t
        new_batch.append(sample)
    return torch.stack(new_batch)

def batch_transform_inverse(new_batch, n_env):
    # new_batch: minibatch_size x n_env * c x t
    c = new_batch.size(1) // n_env
    t = new_batch.shape[2:]
    new_batch = new_batch.reshape(-1, n_env, c, *t)
    batch = []
    for i in range(n_env):
        sample = new_batch[:, i]  # minibatch_size x c x t
        batch.append(sample)
    return torch.cat(batch)  # b x c x t


def set_requires_grad(module, tf=False):
    module.requires_grad = tf
    for param in module.parameters():
        param.requires_grad = tf


def set_no_grad(algorithm, filter_name: list):
    for name, param in algorithm.named_parameters():
        if param.requires_grad and name in filter_name:
            param.requires_grad = False


def set_no_grad_module(algorithm, filter_name: str):
    for name, param in algorithm.named_parameters():
        if param.requires_grad and filter_name in name:
            param.requires_grad = False


def generate_init_value(module, value_clamp):
    for param in module.parameters():
        param.data = torch.rand_like(param)*(2*value_clamp) - value_clamp

def generate_mask(net_a, mask_type="layer", layers=[0]):
    n_params_tot = count_parameters(net_a)
    if mask_type == "layer":
        mask_w = torch.zeros(n_params_tot)
        count = 0
        for name, pa in net_a.named_parameters():
            if any(f"net.{layer}" in name for layer in layers):
                mask_w[count: count + pa.numel()] = 1.
            count += pa.numel()
    elif mask_type == "full":
        mask_w = torch.ones(n_params_tot)
    else:
        raise Exception(f"Unknown mask {mask_type}")
    return mask_w


def generate_theta(net_a, mask_type="layer", layers=[0], delta_theta_clamp=0.1):
    n_params_tot = count_parameters(net_a)
    if mask_type == "layer":
        theta = torch.zeros(n_params_tot)
        count = 0
        for name, pa in net_a.named_parameters():
            if any(f"net.{layer}" in name for layer in layers):
                theta[count: count + pa.numel()] = (torch.rand(pa.numel()).float() * (2 * delta_theta_clamp) - delta_theta_clamp)
            count += pa.numel()
    elif mask_type == "full":
        theta = (torch.rand(n_params_tot).float() * (2 * delta_theta_clamp) - delta_theta_clamp)
    else:
        raise Exception(f"Unknown mask {mask_type}")
    return theta


def count_parameters(model, mode='ind'):
    if mode == 'ind':
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif mode == 'layer':
        return sum(1 for p in model.parameters() if p.requires_grad)
    elif mode == 'row':
        n_mask = 0
        for p in model.parameters():
            if p.dim() == 1:
                n_mask += 1
            else:
                n_mask += p.size(0)
        return n_mask


def get_n_param_layer(net, layers):
    n_param = 0
    for name, p in net.named_parameters():
        if any(f"net.{layer}" in name for layer in layers):
            n_param += p.numel()
    return n_param
