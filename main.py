import logging
import pandas as pd

from torch.utils.data import DataLoader
from gene_dataset import MountGeneExpressionDataset

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
    dataset = MountGeneExpressionDataset(fields, filters, "cases.disease_type", gene_list)
    train_loader = DataLoader(dataset, batch_size=2)
    for idx, data in enumerate(train_loader):
        print(data)
        if idx == 10:
            break


import logging.config
if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    main()
