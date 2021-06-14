import pandas as pd
import torch
import sys
import datetime
import time

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


def speed_test():
    print("1")
    fields, filters, gene_list = initialize_pam50()
    raw_data, label_column = manifest_loader(fields, filters, "cases.disease_type")
    

    # Test case:
#     tcga_base = '/home/grammaright/Downloads/tcga'
    tcga_base = './tcga'
    dataset2 = MountGeneExpressionDataset(tcga_base, raw_data, label_column, gene_list)
    loader2 = DataLoader(dataset2, batch_size=1)
    start = datetime.datetime.now()
    print("2")
    
    for idx, data1 in enumerate(loader2):
        print(idx)
        time.sleep(5)
        if idx > 5:
            break

    end = datetime.datetime.now()
    print('MountGeneExpressionDataset (iter=10, batch_size=1) : {} sec'.format((end - start).seconds))

    print("Test passed")

