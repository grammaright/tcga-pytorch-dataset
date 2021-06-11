import logging
import pandas as pd
import logging.config
import torch

from torch.utils.data import DataLoader
from gene_dataset import NaiveMountGeneExpressionDataset, MountGeneExpressionDataset

logger = logging.getLogger(__name__)


def initialize():
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


def main():
    fields, filters, gene_list = initialize()

    # Test case:
    dataset1 = NaiveMountGeneExpressionDataset(fields, filters, "cases.disease_type", gene_list)
    dataset2 = MountGeneExpressionDataset(fields, filters, "cases.disease_type", gene_list)

    loader1 = DataLoader(dataset1, batch_size=1)
    loader2 = DataLoader(dataset2, batch_size=1)
    for idx, (data1, data2) in enumerate(zip(loader1, loader2)):
        res = torch.equal(data1[0], data2[0])
        if res is False or data1[1] != data2[1]:
            logger.fatal("Diff!!!!!!")
            logger.fatal('{}, {}'.format(data1, data2))
            break
        
        if idx == 10:
            break

    logger.info("Test passed")


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    main()

