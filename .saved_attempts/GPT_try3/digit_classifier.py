import os
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the training and test datasets
train = pd.read_csv('./Train_Test_Data/train.csv')
test = pd.read_csv('./Train_Test_Data/test.csv')import torch
import torch.nn as nn
import torch.nn.functional as F

# Checking if CUDA is available
cuda = torch.cuda.is_available()

class DigitClassifierCNN(nn.Module):
    def __init__(self):
        super(DigitClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output
