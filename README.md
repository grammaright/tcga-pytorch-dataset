# PyTorch DataSet for TCGA

## Introduction

TCGA is huge (almost 2.5 petabytes) and unstructured data.
Downloading all data is impossible, so a data scientist generally mounts the AWS-providing S3 bucket to use the data.
However, using the mounted one is too slow with a naive PyTorch DataSet.

We introduce a customized DataSet (based on PyTorch IterableDataset) which provides faster performance.
We accelerate the Dataset by pipelining (hiding download latency), scheduling (small data first), and caching (holding cost-effective and evicting cost-ineffective).

## How to use?

### preliminaries

To use our DataLoader, you need to do the following.

1. Install goofys and mount `tcga-2-open`.
    - goofys URL: https://github.com/kahing/goofys
    - Mount command line: `mkdir tcga; goofys tcga-2-open tcga;`.
2. Install the python packages described in `requirements.txt`
    - Of course you can install by `pip install -r requirements.txt`.

### Usage

We provies `MountGeneExpressionDataset` and `NaiveMountGeneExpressionDataset` which are custom `Dataset` class of torch.
You just need two additional lines and `MountGeneExpressionDataset` instead of other Dataset.
The following is an example:

```python
fields, filters, gene_list = initialize_pan_cancer()
raw_data, label_column = manifest_loader(fields, filters, "cases.disease_type")
dataset = MountGeneExpressionDataset('./tcga', raw_data, label_column, gene_list)   # The first parameter is the mount point.
loader = DataLoader(dataset, batch_size=1)
for data in loader:
    # Your ML tasks here!
    pass
```

You can also test the Dataloaders.
See the next step.

### Tests

We provies three test cases.
1. Equality test: `NaiveMountGeneExpressionDataset` and `MountGeneExpressionDataset` produces the same results? See `eq_test.py`.
2. Speed test: Is `MountGeneExpressionDataset` really fast than the naive one? See `sp_test.py`.
3. NN test: Is `MountGeneExpressionDataset` really works in NN task? See `nn_test.py`.


## Benchmark

TBA
