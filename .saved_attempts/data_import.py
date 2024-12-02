import os
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the dataset
df = pd.read_csv('./Train_Test_Data/dataset.csv')
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

# Set learning rate and batch size
learning_rate = 0.001
batch_size = 64

# Convert the dataset into a DataLoader for batch processing
data_loader = torch.utils.data.DataLoader(df, batch_size=batch_size, shuffle=True)

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Define an optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, data in enumerate(data_loader):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = net(inputs)

        # Calculate loss
        loss = loss_fn(outputs, labels)

        # Backward propagation and optimization
        loss.backward()
        optimizer.step()