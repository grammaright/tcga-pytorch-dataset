import logging
import pandas as pd
import logging.config
import torch
import sys
import datetime
import time

from torch.utils.data import DataLoader
from gene_dataset import NaiveMountGeneExpressionDataset, MountGeneExpressionDataset

from gene_dataset import NaiveMountGeneExpressionDataset, MountGeneExpressionDataset, manifest_loader

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_pam50():
    gene_list = list(pd.read_csv("./pam50.txt",sep="\t", header = None, index_col = False)[0])

    fields = [
        "file_name",
        "cases.submitter_id",
        "cases.samples.sample_type",
        "cases.disease_type",
        "file_id"
    ]

    fields = ",".join(fields)

    # This set of filters is nested under an 'and' operator.
    filters = {
        "op": "and",
        "content":[
            {
                "op": "in",
                "content":{
                    "field": "cases.project.program.name",
                    "value": ["TCGA"]
                }
            },
            {
                "op": "in",
                "content":{
                    "field": "cases.project.primary_site",
                    "value": ["Breast", "Lung"]
                }
            },
            {
                "op": "in",
                "content":{
                    "field": "files.data_category",
                    "value": ["Gene expression"]
                }
            },
            {
                "op": "in",
                "content":{
                    "field": "files.data_type",
                    "value": ["Gene expression quantification"]
                }
            },
            {
                "op": "in",
                "content":{
                    "field": "files.experimental_strategy",
                    "value": ["RNA-Seq"]
                }
            },
            {
                "op": "in",
                "content":{
                    "field": "files.data_format",
                    "value": ["TXT"]
                }
            }
        ]
    }

    return fields, filters, gene_list


def speed_test():
    fields, filters, gene_list = initialize_pam50()
    raw_data, label_column = manifest_loader(fields, filters, "cases.disease_type")

    # Test case:
    dataset1 = NaiveMountGeneExpressionDataset(raw_data, label_column, gene_list)
    loader1 = DataLoader(dataset1, batch_size=1)
    start = datetime.datetime.now()
    for idx, data1 in enumerate(loader1):
        logger.info(idx)
        time.sleep(5)
        if idx > 10:
            break

    end = datetime.datetime.now()
    logger.info('NaiveMountGeneExpressionDataset (iter=10, batch_size=1) : {} sec'.format((end - start).seconds))

    dataset2 = MountGeneExpressionDataset(raw_data, label_column, gene_list)
    loader2 = DataLoader(dataset2, batch_size=1)
    start = datetime.datetime.now()
    for idx, data1 in enumerate(loader2):
        logger.info(idx)
        time.sleep(5)
        if idx > 10:
            break

    end = datetime.datetime.now()
    logger.info('MountGeneExpressionDataset (iter=10, batch_size=1) : {} sec'.format((end - start).seconds))

    logger.info("Test passed")

