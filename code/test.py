import torch
from cnnArchitecture import CNN # Replace with your actual model class

# Load the model from the .pth file
model = CNN(100, 1000).to('cpu')
model.load_state_dict(torch.load('D:\\Donnees\\Documents\\insa\\5IF\\PSAT\\CNN1D\\Analyse-de-conscience\\code\\best_model_checkpoint_0_4.pth'))

# Alternatively, you can print the architecture layer by layer
for name, param in model.named_parameters():
    print(f"Layer: {name}, Size: {param.size()}")
