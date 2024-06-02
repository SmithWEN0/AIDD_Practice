import torch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

from torch.autograd import Variable
import random

import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader

class prediction_model(nn.Module):

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(prediction_model, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden1_size)
        # self.dropout1 = nn.Dropout(p=0.1)
        self.batch1 = nn.BatchNorm1d(hidden1_size)

        self.hidden2 = nn.Linear(hidden1_size, hidden2_size)
        # self.dropout2 = nn.Dropout(p=0.2)
        self.batch2 = nn.BatchNorm1d(hidden2_size)

        # self.hidden3 = nn.Linear(hidden2_size, hidden3_size)
        # self.batch3 = nn.BatchNorm1d(hidden3_size)

        self.predict = nn.Linear(hidden2_size, output_size)
    
    def forward(self, input):
        result = self.hidden1(input)
        # result = self.dropout1(result)
        result = self.batch1(result)

        result = F.leaky_relu(result)

        result = self.hidden2(result)
        # result = self.dropout2(result)
        result = self.batch2(result)
        result = F.leaky_relu(result)

        # result = self.hidden3(result)
        # result = self.batch3(result)
        # result = F.leaky_relu(result)

        result = self.predict(result)

        return result