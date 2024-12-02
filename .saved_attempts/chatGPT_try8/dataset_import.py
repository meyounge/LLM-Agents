import os
import pandas as pd

# Ensuring that the home directory is being used
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the datasets
def load_datasets():
    train_data = pd.read_csv('./Train_Test_Data/train.csv')
    test_data = pd.read_csv('./Train_Test_Data/test.csv')
    return train_data, test_data# Importing necessary libraries\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.autograd import Variable\n\n# Checking for GPU\ndevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')\n\n# Defining the CNN\nclass Net(nn.Module):\n    def __init__(self):\n        super(Net, self).__init__()\n        self.conv1 = nn.Conv2d(1, 32, 3, 1)\n        self.conv2 = nn.Conv2d(32, 64, 3, 1)\n        self.dropout1 = nn.Dropout2d(0.25)\n        self.dropout2 = nn.Dropout2d(0.5)\n        self.fc1 = nn.Linear(9216, 128)\n        self.fc2 = nn.Linear(128, 10)\n\n    def forward(self, x):\n        x = self.conv1(x)\n        x = torch.relu(x)\n        x = self.conv2(x)\n        x = torch.relu(x)\n        x = torch.max_pool2d(x, 2)\n        x = self.dropout1(x)\n        x = torch.flatten(x, 1)\n        x = self.fc1(x)\n        x = torch.relu(x)\n        x = self.fc2(x)\n        output = torch.log_softmax(x, dim=1)\n        return output\n\n# Creating an instance of the CNN and moving it to the GPU\nnet = Net().to(device)\n\n# Defining the loss function and optimizer\nloss_function = nn.CrossEntropyLoss()\noptimizer = torch.optim.Adam(net.parameters())

# Training parameters
alpha = 0.001
batch_size = 64
n_epochs = 10

# Loading the datasets
train_data, test_data = load_datasets()

# Training loop
for epoch in range(n_epochs):
    for i in range(0, len(train_data), batch_size):
        # Prepare batch
        batch_X = train_data[i:i+batch_size].view(-1, 1, 28, 28).to(device)
        batch_y = test_data[i:i+batch_size].to(device)

        # Zero the parameter gradients
        net.zero_grad()

        # Forward + backward + optimize
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

print('Finished Training')