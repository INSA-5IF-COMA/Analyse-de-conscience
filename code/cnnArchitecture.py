import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_dim, seq_dim, dropout_rate=0.7):
        super(CNN, self).__init__()
        
        # Define the architecture with layers based on the input arguments
        self.conv1 = nn.Conv1d(seq_dim, 16, 6)
        self.conv2 = nn.Conv1d(16, 64, 3)
        self.conv3 = nn.Conv1d(64, 32 , 3)
        self.relu = nn.ReLU()

        
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # Fully connected layer
        self.fc_input_size = 32 * ( input_dim - 6 - 3 - 3 + 3) 
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2) 

    def forward(self, x):
        # Forward pass
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.dropout1(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x