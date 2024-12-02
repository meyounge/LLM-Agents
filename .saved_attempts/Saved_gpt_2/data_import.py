import pandas as pd
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the data
train_data = pd.read_csv('./Train_Test_Data/train.csv')
test_data = pd.read_csv('./Train_Test_Data/test.csv')

print('Data imported successfully.')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Check if GPU is available
use_cuda = torch.cuda.is_available()

def CNN():
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=5, padding=2),
        nn.MaxPool2d(2),
        nn.ReLU(True),
        nn.Conv2d(32, 64, kernel_size=5, padding=2),
        nn.MaxPool2d(2),
        nn.ReLU(True),
        nn.Flatten(),
        nn.Linear(7*7*64, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 10),
    )
    if use_cuda:
        model = model.cuda()
    return model

model = CNN()
print('Model created successfully.')
# Set hyperparameters
learning_rate = 0.001
batch_size = 100
num_epochs = 5

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        images = Variable(batch.view(-1, 1, 28, 28))
        labels = Variable(batch.labels)

        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    print ('Epoch [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, loss.data))

print('Training completed successfully.')