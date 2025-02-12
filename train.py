import torch
import argparse
from model import build_model, save_checkpoint
from utils import load_data

def train_model(data_dir, arch, lr, hidden_units, epochs, save_dir, gpu):
    # Load data
    dataloaders, image_datasets = load_data(data_dir)
    
    # Build model
    model, criterion, optimizer = build_model(arch, hidden_units, lr, gpu)
    
    # Train model
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            optimizer.step()
            loss = criterion(outputs, labels)
            loss.backward()
            
            train_loss += loss.item()
        
        print(f"Epoch {epoch+1}.. Training Loss: {train_loss/len(dataloaders['train']):.4f}")

    # Save checkpoint
    save_checkpoint(model, image_datasets, arch, hidden_units, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Dataset directory")
    parser.add_argument("--save_dir", type=str, default=".", help="Checkpoint save directory")
    parser.add_argument("--arch", type=str, default="resnet34", help="Model architecture (resnet34, vgg16, etc.)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Hidden units in classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")

    args = parser.parse_args()
    train_model(args.data_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.save_dir, args.gpu)
