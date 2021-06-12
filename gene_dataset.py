import logging
import os
import sys

import pandas as pd
import torch
import requests

from torch.utils.data import IterableDataset
from io import StringIO
from mp import DatasetMPManager


logging.basicConfig(filename='gene_dataset.log', filemode='a', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
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
    def __init__(self, tcga_base, raw_data, label_column, gene_list, normalized=False):
        super(NaiveMountGeneExpressionDataset).__init__()
        ## 지정해준 fields, filter에 해당하는 파일들의 목록(manifest) 를 불러온다
        ## 어떤 coulmn을 label로 사용할 것인지 label_column을 통해 지정해주어야 한다. 
        raw_data.rename(lambda x : x.split('.')[-1], axis='columns', inplace = True)
        label_column = label_column.split('.')[-1]
        self.file_list = list(raw_data['file_id'])
        self.label_list = list(raw_data[label_column])
        self.label_dict = {}
        for idx, label in enumerate(set(self.label_list)):
            self.label_dict[label] = idx
        
        self.gene_list = gene_list
        self.normalized = normalized
        self.tcga_base = tcga_base

    def __iter__(self):
        for filename, label in zip(self.file_list, self.label_list):
            raw_data = pd.read_csv(self.tcga_base + '/' + filename + "/" + os.listdir(self.tcga_base + '/' + filename)[0], sep='\t')
            logger.info('naive target file={}'.format(self.tcga_base + '/' + filename + "/" + os.listdir(self.tcga_base + '/' + filename)[0]))
            label_idx = self.label_dict[label]
            gene_column, count_column = '', ''

            if 'gene_id' in list(raw_data.columns):
                gene_column = 'gene_id'
            elif 'gene' in list(raw_data.columns):
                gene_column = 'gene'
                
            if not self.normalized:
                if 'raw_count' in list(raw_data.columns):
                    count_column = 'raw_count'
                elif 'raw_counts' in list(raw_data.columns):
                    count_column = 'raw_counts'
            else :
                count_column = 'normalized_count'

            try:
                raw_data[gene_column] = raw_data[gene_column].apply(lambda x: x.split('|')[0])
                #gene_id는 [symbol|PubMedID] (예 BRCA1|11331580) 의 형식으로 기재되어 있음
                #Symbol만을 남김
                raw_data = raw_data[raw_data[gene_column].isin(self.gene_list)]
                expression = raw_data[count_column].values
                logger.info('{} yield'.format(filename))
                yield (torch.tensor(expression) ,label_idx)
            except Exception as e:
                logger.info('skip')
                #print(e)
                continue


class MountGeneExpressionDataset(IterableDataset):
    def __init__(self, tcga_base, raw_data, label_column, gene_list, normalized=False):
        super(MountGeneExpressionDataset).__init__()
        # raw_data, label_column = manifest_loader(fields, filters,label)
        ## 지정해준 fields, filter에 해당하는 파일들의 목록(manifest) 를 불러온다
        ## 어떤 coulmn을 label로 사용할 것인지 label_column을 통해 지정해주어야 한다. 
        raw_data.rename(lambda x : x.split('.')[-1], axis='columns', inplace = True)
        label_column = label_column.split('.')[-1]
        self.file_list = list(raw_data['file_id'])
        self.label_list = list(raw_data[label_column])
        self.label_dict = {}
        for idx, label in enumerate(set(self.label_list)):
            self.label_dict[label] = idx
        
        self.gene_list = gene_list
        self.normalized = normalized

        # spwan
        self.manager = DatasetMPManager(self.file_list, tcga_base)

    def __iter__(self):
        for filename, label in zip(self.file_list, self.label_list):
            raw_data = self.manager.next()
            label_idx = self.label_dict[label]
            gene_column, count_column = '', ''

            if 'gene_id' in list(raw_data.columns):
                gene_column = 'gene_id'
            elif 'gene' in list(raw_data.columns):
                gene_column = 'gene'
                
            if not self.normalized:
                if 'raw_count' in list(raw_data.columns):
                    count_column = 'raw_count'
                elif 'raw_counts' in list(raw_data.columns):
                    count_column = 'raw_counts'
            else :
                count_column = 'normalized_count'

            try:
                raw_data[gene_column] = raw_data[gene_column].apply(lambda x: x.split('|')[0])
                #gene_id는 [symbol|PubMedID] (예 BRCA1|11331580) 의 형식으로 기재되어 있음
                #Symbol만을 남김
                raw_data = raw_data[raw_data[gene_column].isin(self.gene_list)]
                expression = raw_data[count_column].values
                logger.info('yield')
                yield (torch.tensor(expression), label_idx)
            except Exception as e:
                logger.warning('__iter__(): exception = {}'.format(str(e)))
                continue

