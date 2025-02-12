import torch
import argparse
import json
from model import load_checkpoint
from utils import process_image
from PIL import Image
import numpy as np

def predict(image_path, checkpoint_path, top_k, category_names, gpu):
    # Load model
    model = load_checkpoint(checkpoint_path)
    
    # Process image
    image = process_image(image_path)
    image = image.unsqueeze(0)

    # Move to device
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    image = image.to(device)

    # Get predictions
    with torch.no_grad():
        output = model(image)
    
    top_probs, top_indices = torch.exp(output).topk(top_k)
    top_probs, top_indices = top_probs.cpu().numpy()[0], top_indices.cpu().numpy()[0]
    
    # Map class index to label
    class_to_idx = model.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_indices]

    # Map class labels to flower names
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_classes = [cat_to_name[label] for label in top_classes]

    return top_probs, top_classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to image")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K predictions")
    parser.add_argument("--category_names", type=str, help="JSON file mapping labels to names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")

    args = parser.parse_args()
    probs, labels = predict(args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu)

    for prob, label in zip(probs, labels):
        print(f"{label}: {prob*100:.2f}%")
