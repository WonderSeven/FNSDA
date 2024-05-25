

import math

from torch.optim.lr_scheduler import LambdaLR


# ====================================================================
# Cosine_with_warmup
# ====================================================================
class CosineLRLambda(object):
    def __init__(self, num_warmup_steps, num_training_steps, num_cycles):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles

    def __call__(self, current_step):  # the function formerly known as "bar"
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / \
            float(max(1, self.num_training_steps - self.num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))


class CosineWithWarmupScheduler(LambdaLR):
    def __init__(self, optimizer, num_warmup_steps: int,
                 num_training_steps: int, num_cycles: float = 0.5,
                 last_epoch=-1, verbose=False):
        lr_lambda = CosineLRLambda(
            num_warmup_steps, num_training_steps, num_cycles)
        super().__init__(optimizer, lr_lambda, last_epoch, verbose)


# ====================================================================
# Exponential_with_warmup
# ====================================================================
class ExponentialLRLambda(object):
    def __init__(self, num_warmup_steps, gamma):
        self.num_warmup_steps = num_warmup_steps
        self.gamma = gamma

    def __call__(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))

        progress = current_step - self.num_warmup_steps
        return self.gamma**progress


class ExponentialWithWarmupScheduler(LambdaLR):
    def __init__(self, optimizer, num_warmup_steps: int,
                 gamma=0.9999, last_epoch=-1, verbose=False):
        lr_lambda = ExponentialLRLambda(num_warmup_steps, gamma)
        super().__init__(optimizer, lr_lambda, last_epoch, verbose)


# ====================================================================
# Linear_with_warmup
# ====================================================================
class LinearLRLambda:
    def __init__(self, num_warmup_steps, num_training_steps):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

    def __call__(self, current_step):  # the function formerly known as "bar"
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return max(0.0, float(self.num_training_steps - current_step) /
                   float(max(1, self.num_training_steps - self.num_warmup_steps)))


class LinearWithWarmupScheduler(LambdaLR):
    def __init__(self, optimizer, num_warmup_steps: int,
                 num_training_steps: int, last_epoch=-1, verbose=False):

        lr_lambda = LinearLRLambda(num_warmup_steps, num_training_steps)
        super().__init__(optimizer, lr_lambda, last_epoch, verbose)
