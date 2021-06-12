import logging
import pandas as pd
import logging.config
import torch
import sys

from torch.utils.data import DataLoader
from gene_dataset import NaiveMountGeneExpressionDataset, MountGeneExpressionDataset

from nn_test import nn_test
from eq_test import equal_test
from sp_test import speed_test

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # equal_test()
    speed_test()
    # nn_test()


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    main()

