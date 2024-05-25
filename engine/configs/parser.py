"""
Config
"""

import os
import pdb
import sys
import yaml
import argparse
from typing import Dict
from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)

def default_config() -> CN:
    """
    Get a yacs CfgNode object with the default config values.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

class BaseOptions(object):
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """
    def __init__(self, opts):
        """Reset the class; indicates the class hasn't been initialized"""
        self.opts = opts
        self.config = None
        self.opts = self.initialize()

    def _parse_args(self, opts) -> Dict:
        return vars(opts)

    def save_config(self, cfg, savepath):
        with open(savepath, 'w') as f:
            f.write(cfg.dump())


    def get_config(self, config_file: str, merge: bool = True) -> CN:
        """
        Read a config file and optionally merge it with the default config file.
        Args:
          config_file (str): Path to config file.
          merge (bool): Whether to merge with the default config or not.
        Returns:
          CfgNode: Config as a yacs CfgNode object.
        """
        if merge:
            cfg = default_config()
        else:
            cfg = CN(new_allowed=True)
        cfg.merge_from_file(config_file)
        cfg.freeze()
        return cfg

    def initialize(self):
        cfg_argparse = self._parse_args(self.opts)
        config_file = cfg_argparse['config_file']

        cfg = default_config()
        cfg.merge_from_file(config_file)
        cfg.update(cfg_argparse)
        cfg.freeze()
        # print(cfg.dump())

        if cfg.trainer.record:
            output_dir = cfg['trainer']['output_dir']
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            _savepath = os.path.join(output_dir, 'config.yaml')
            self.save_config(cfg, _savepath)
        return cfg


def get_config(config_yaml):
    with open(config_yaml, 'r') as stream:
        config = yaml.safe_load(stream)
        # config['gpu_ids'] = parse_gpu_ids(config['gpu_ids'])
        return config
