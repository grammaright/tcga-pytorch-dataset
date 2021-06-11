import logging
import os
import sys

import pandas as pd
import torch
import requests

from torch.utils.data import IterableDataset
from io import StringIO
from mp import DatasetMPManager


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

files_endpt = "https://api.gdc.cancer.gov/legacy/files"


def manifest_loader(fields, filters, label):
    assert label in fields
    params = {
        "filters": filters,
        "fields": fields,
        "format": "TSV",
        "size": "20000"
    }

    response = requests.post(files_endpt, headers={"Content-Type": "application/json"}, json=params)
    manifest = response.content

    data = StringIO(str(response.content, 'utf-8'))
    manifest = pd.read_csv(data, sep='\t', index_col=False)

    return manifest, label


class NaiveMountGeneExpressionDataset(IterableDataset):
    def __init__(self, fields, filters, label, gene_list):
        super(NaiveMountGeneExpressionDataset).__init__()
        raw_data, label_column = manifest_loader(fields, filters, label)
        raw_data.rename(lambda x: x.split('.')[-1], axis='columns', inplace=True)
        label_column = label_column.split('.')[-1]
        self.file_list = list(raw_data['file_id'])
        self.label_list = list(raw_data[label_column])
        self.label_dict = {}
        for idx, label in enumerate(set(self.label_list)):
            self.label_dict[label] = idx
        self.gene_list = gene_list

    def __iter__(self):
        for filename, label in zip(self.file_list, self.label_list):

            raw_data = pd.read_csv("./tcga3/" + filename + "/" + os.listdir("./tcga3/" + filename)[0], sep='\t')
            label_idx = self.label_dict[label]
            try:
                raw_data['gene_id'] = raw_data['gene_id'].apply(lambda x: x.split('|')[0])
                raw_data[raw_data['gene_id'].isin(self.gene_list)]
                yield (torch.Tensor(list(raw_data['raw_count'])), label_idx)
            except Exception as e:
                continue


class MountGeneExpressionDataset(IterableDataset):
    def __init__(self, fields, filters, label, gene_list):
        super(MountGeneExpressionDataset).__init__()
        raw_data, label_column = manifest_loader(fields, filters, label)
        raw_data.rename(lambda x: x.split('.')[-1], axis='columns', inplace=True)
        label_column = label_column.split('.')[-1]
        self.file_list = list(raw_data['file_id'])
        self.label_list = list(raw_data[label_column])
        self.label_dict = {}
        for idx, label in enumerate(set(self.label_list)):
            self.label_dict[label] = idx
        self.gene_list = gene_list

        # spwan
        self.manager = DatasetMPManager(self.file_list)

    def __iter__(self):
        for label in self.label_list:           # hitting manager.next() and label_list must be done with the same order
            label_idx = self.label_dict[label]
            try:
                raw_data = self.manager.next()
                raw_data['gene_id'] = raw_data['gene_id'].apply(lambda x: x.split('|')[0])
                raw_data[raw_data['gene_id'].isin(self.gene_list)]
                yield (torch.Tensor(list(raw_data['raw_count'])), label_idx)
            except Exception as e:
                logger.warning('__iter__(): exception = {}'.format(str(e)))
                continue

