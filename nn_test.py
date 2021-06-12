import logging
import pandas as pd
import logging.config
import torch
import torch.nn.functional as F
import time
import sys
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from gene_dataset import NaiveMountGeneExpressionDataset, MountGeneExpressionDataset, manifest_loader


logging.basicConfig(filename='gene_dataset.log', filemode='a', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_pan_cancer():
    gene_list = list(pd.read_excel("./pan_cancer.xlsx", header = None, index_col = False)[0])

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


class DiseaseClassification(nn.Module):
    def __init__(self, input_dim ,hidden_dim, output_dim):
        
        super().__init__()
        self.fc1 = nn.Linear(input_dim,500)
        self.fc2 = nn.Linear(500,1000)
        self.fc3 = nn.Linear(1000,2000)
        self.fc4 = nn.Linear(2000,1000)
        self.fc5 = nn.Linear(1000,500)
        self.fc6 = nn.Linear(500, output_dim)
        
    def forward(self,x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        
        return x


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) 
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for expression, label in iterator:
        
        logger.info('start_batch_train')
        optimizer.zero_grad()

        predictions = model(expression)

        loss = criterion(predictions, label)
                
        acc = categorical_accuracy(predictions, label)
        
        loss.backward()
        
        optimizer.step()
        logger.info('end_batch_train')
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for expression, label in iterator:

            predictions = model(expression)
        
            loss = criterion(predictions, label)  
            acc = categorical_accuracy(predictions, label)
        
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def nn_test():
    fields, filters, gene_list = initialize_pan_cancer()
    raw_data, label_column = manifest_loader(fields, filters, "cases.disease_type")
    train_data, test_data = train_test_split(raw_data, test_size=0.33, random_state=42)
    tcga_base = '/home/grammaright/Downloads/tcga'

    train_dataset = MountGeneExpressionDataset(tcga_base, train_data, label_column, gene_list)
    test_dataset = MountGeneExpressionDataset(tcga_base, train_data, label_column, gene_list)
    # test_dataset = MountGeneExpressionDataset(test_data, label_column, gene_list)
    train_loader = DataLoader(train_dataset, batch_size = 10)
    test_loader = DataLoader(test_dataset, batch_size = 10)

    INPUT_SIZE = 1346
    OUTPUT_SIZE = 3
    HIDDEN_SIZE = 500
    criterion = nn.CrossEntropyLoss()
    model = DiseaseClassification(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    optimizer = optim.Adam(model.parameters(), lr = 0.02)
    N_EPOCHS = 10
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, train_loader, optimizer, criterion)
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


