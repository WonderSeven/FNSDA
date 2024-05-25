import pdb
import sys
import yaml
import argparse
from PIL import ImageFile
sys.dont_write_bytecode = True
ImageFile.LOAD_TRUNCATED_IMAGES = True

from engine import trainer
from engine.configs.parser import BaseOptions


def parse_args():
    parser = argparse.ArgumentParser(description='General Physical Simulation')
    parser.add_argument('--config_file', type=str, default='./configs/ERM.yaml')
    opts = parser.parse_args()
    return opts


if __name__ == '__main__':
    if sys.version_info<(3,7,0):
        sys.stderr.write("You need python 3.7 or later to run this script.\n")
        sys.exit(1)

    opts = parse_args()
    cfg = BaseOptions(opts).opts
    trainer = trainer.Trainer(cfg)
    trainer.train()
