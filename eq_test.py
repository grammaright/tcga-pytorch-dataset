import pandas as pd
import logging.config
import torch
import sys

from torch.utils.data import DataLoader
from gene_dataset import NaiveMountGeneExpressionDataset, MountGeneExpressionDataset

from gene_dataset import NaiveMountGeneExpressionDataset, MountGeneExpressionDataset, manifest_loader


def initialize_pam50():
    gene_list = list(pd.read_csv("./pam50.txt",sep="\t", header = None, index_col = False)[0])

    fields = [
        "file_name",
        "cases.submitter_id",
        "cases.samples.sample_type",
        "cases.disease_type",
        "file_id",
        "file_size"
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


def equal_test():
    fields, filters, gene_list = initialize_pam50()
    raw_data, label_column = manifest_loader(fields, filters, "cases.disease_type")
    tcga_base = '/home/grammaright/Downloads/tcga'

    # Test case:
    dataset1 = NaiveMountGeneExpressionDataset(tcga_base, raw_data, label_column, gene_list)
    dataset2 = MountGeneExpressionDataset(tcga_base, raw_data, label_column, gene_list)

    loader1 = DataLoader(dataset1, batch_size=1)
    loader2 = DataLoader(dataset2, batch_size=1)
    for idx, (data1, data2) in enumerate(zip(loader1, loader2)):
        print(idx, data1[0], data2[0])
        res = torch.equal(data1[0], data2[0])
        if res is False or data1[1] != data2[1]:
            print('Diff!!!!!')
            break

    print('Test passed')

