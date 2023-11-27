import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_dim, seq_dim):
        super(CNN, self).__init__()
        
        # Define the architecture with layers based on the input arguments
        self.conv1 = nn.Conv1d(seq_dim, 32, 5)
        self.conv2 = nn.Conv1d(32, 64, 3)
        self.conv3 = nn.Conv1d(64, 128, 3)
        self.conv4 = nn.Conv1d(128, 256,3)
        self.relu = nn.ReLU()
        
        # Mise à jour de fc_input_size en fonction des couches de pooling
        self.fc_input_size = 256 * ( input_dim - 5 - 3 - 3 - 3 + 4) 
        self.fc = nn.Linear(self.fc_input_size, 2)

    def forward(self, x):
        # Forward pass
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.relu(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        
        return x