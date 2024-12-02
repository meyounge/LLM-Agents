
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hyperparameters
learning_rate = 0.001
batch_size = 64

# Loss function
criterion = nn.CrossEntropyLoss()

# DataLoader
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# Initialize the CNN
model = CNN()  

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train(model, train_loader):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (batch_idx+1) % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Step {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}')

train(model, train_loader)
