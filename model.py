import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

def build_model(arch, hidden_units, lr, gpu):
    if arch == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        input_size = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    
    # Set optimizer and loss function
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    return model, criterion, optimizer

def save_checkpoint(model, image_datasets, arch, hidden_units, save_dir):
    checkpoint = {
        'arch': arch,
        'class_to_idx': image_datasets['train'].class_to_idx,
        'model_state_dict': model.state_dict(),
        'hidden_units': hidden_units
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model, _, _ = build_model(checkpoint['arch'], checkpoint['hidden_units'], 0.001, False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
