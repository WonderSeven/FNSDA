import copy
import random
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices, minibatch_size=2):
        self.minibatch_size = minibatch_size
        if not any(isinstance(el, list) for el in indices):
            self.indices = [indices]
        else:
            self.indices = indices
        self.env_len = len(self.indices[0])

    def __iter__(self):
        if len(self.indices) > 1:
            l_indices = copy.deepcopy(self.indices)

            l_iter = list()
            for _ in range(0, self.env_len, self.minibatch_size):
                for i in range(len(l_indices)):
                    l_iter.extend(l_indices[i][:self.minibatch_size])
                    del l_indices[i][:self.minibatch_size]
        else:
            l_iter = copy.deepcopy(self.indices[0])
        return iter(l_iter)

    def __len__(self):
        return sum([len(el) for el in self.indices])


class SubsetRamdomSampler(Sampler):
    def __init__(self, indices, minibatch_size=2, same_order_in_groups=True):
        self.minibatch_size = minibatch_size
        self.same_order_in_groups = same_order_in_groups
        if not any(isinstance(el, list) for el in indices):
            self.indices = [indices]
        else:
            self.indices = indices
        self.env_len = len(self.indices[0])

    def __iter__(self):
        if len(self.indices) > 1:
            if self.same_order_in_groups:
                l_shuffled = copy.deepcopy(self.indices)
                random.shuffle(l_shuffled[0])
                for i in range(1, len(self.indices)):
                    l_shuffled[i] = [el + i * self.env_len for el in l_shuffled[0]]
            else:
                l_shuffled = copy.deepcopy(self.indices)
                for l in l_shuffled:
                    random.shuffle(l)

            l_iter = list()
            for _ in range(0, self.env_len, self.minibatch_size):
                for i in range(len(l_shuffled)):
                    l_iter.extend(l_shuffled[i][:self.minibatch_size])
                    del l_shuffled[i][:self.minibatch_size]
        else:
            l_shuffled = copy.deepcopy(self.indices[0])
            random.shuffle(l_shuffled)
            l_iter = l_shuffled
        return iter(l_iter)

    def __len__(self):
        return sum([len(el) for el in self.indices])


def DataLoaderODE(dataset, minibatch_size, n_env, is_train=True):
    if is_train:
        sampler = SubsetRamdomSampler(indices=dataset.indices, minibatch_size=minibatch_size)
    else:
        sampler = SubsetSequentialSampler(indices=dataset.indices, minibatch_size=minibatch_size)
    dataloader_params = {
        'dataset': dataset,
        'batch_size': minibatch_size * n_env,
        'num_workers': 0,
        'sampler': sampler,
        'pin_memory': True,
        'drop_last': False
    }
    return DataLoader(**dataloader_params)
