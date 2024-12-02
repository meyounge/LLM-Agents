import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

# To ensure that the script runs in the correct directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Loading the train and test data
class DigitDataset(Dataset):
    def __init__(self, file_path, is_test=False):
        self.data = pd.read_csv(file_path)
        self.is_test = is_test
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if not self.is_test:
            image = self.data.iloc[index, 1:].values.astype('uint8').reshape((28, 28))
            label = self.data.iloc[index, 0]
            return self.transform(image), label
        else:
            image = self.data.iloc[index, :].values.astype('uint8').reshape((28, 28))
            return self.transform(image)

# Parameters
batch_size = 64
learning_rate = 0.001

# Dataloaders
train_data = DigitDataset('./Train_Test_Data/train.csv')
test_data = DigitDataset('./Train_Test_Data/test.csv', is_test=True)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# CNN Model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = ConvNet()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
num_epochs = 10

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backprop and perform Adam optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

print("Training complete")

# Save the model
torch.save(model.state_dict(), 'model.ckpt')
