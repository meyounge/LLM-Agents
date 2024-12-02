import os
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

train_data = pd.read_csv('./Train_Test_Data/train.csv')
test_data = pd.read_csv('./Train_Test_Data/test.csv')

import torch
import torch.nn as nn
import torch.optim as optim

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Convolutional Neural Network

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Initialize the network and optimizer
model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# Defining training parameters
learning_rate = 0.001
batch_size = 64
loss_fn = torch.nn.CrossEntropyLoss()

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

# Print for debug
print('Creating train_loader...')
try:
    dataset = some_function_to_define_dataset()
except Exception as e:
    print('Error while creating dataset:', str(e))

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)