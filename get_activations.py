import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import logging
import json
import random

import numpy as np
import json
from tqdm.notebook import tqdm

import argparse

from PIL import Image
import requests
import matplotlib.pyplot as plt

from transformers import CLIPProcessor, CLIPModel

import pandas as pd

activations_list = []

# Path to your imagenet_class.json file
json_file_path = '/home/mila/s/sonia.joseph/ViT-Planetarium/my_draft/test_nbs/imagenet_class_index.json'
imagenet_path = '/network/datasets/imagenet.var/imagenet_torchvision/val/'

# Load the JSON file into a Python dictionary
with open(json_file_path, 'r') as file:
    num_to_word_dict = json.load(file)

# Create a reverse dictionary for word to number mapping
word_to_num_dict = {}
for num, words in num_to_word_dict.items():
    for word in words:  # Assuming each entry in num_to_word_dict is a list of words
        word_to_num_dict[word] = num

# Function to get the class name from a label
def get_class_name(label):
    # Assuming the label maps to a list of class names
    return num_to_word_dict.get(str(label), ["Unknown label"])[1]

# Function to get the label from a class name
def get_label(class_name):
    return word_to_num_dict.get(class_name, "Unknown class name")

# Get class names
imagenet_class_nums = np.arange(0, 1000, 1)
imagenet_class_names = ["{}".format(get_class_name(i)) for i in imagenet_class_nums]


# Function to load images based on saved order
def load_images_in_order(indices_path, imagenet_dataset):
    indices = np.load(indices_path)
    subset_dataset = Subset(imagenet_dataset, indices)
    data_loader = DataLoader(subset_dataset, batch_size=1)
    return data_loader

# Load the ImageNet dataset

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

batch_size = 1
imagenet_data = datasets.ImageFolder(imagenet_path, transform=data_transforms)
data_loader = load_images_in_order('imagenet_sample_indices.npy', imagenet_data)


# Function to register the hook
def register_hook(module):
    def hook(module, input, output):
        activations_list.append(output.detach())
    return module.register_forward_hook(hook)

def process_images(model, processor, total_images, total_labels, batch_idx):
    detailed_activations = []

    # Takes in batch size = 1 at a time.
    for i, (images, labels) in enumerate(zip(total_images, total_labels)):
        activations_list.clear()

        inputs = processor(text=imagenet_class_names, images=images, return_tensors="pt", padding=True)
        outputs = model(**inputs)

        class_name = get_class_name(labels.item())
        
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        predicted_class_name = imagenet_class_names[probs.argmax(dim=1)]

        # Assuming batch size of 1 for simplicity.
        # Enumerate through patches and neurons to get each value
        for patch_idx, patch_activations in enumerate(activations_list[0][0]): # Get first batch
            for neuron_idx, activation_value in enumerate(patch_activations):
                detailed_activations.append({
                    'batch_idx': batch_idx,
                    # 'image_idx': i,
                    'class_name': class_name,
                    'predicted': predicted_class_name,
                    'patch_idx': patch_idx,
                    'neuron_idx': neuron_idx,
                    'activation_value': activation_value.item()
                })


    return detailed_activations

def main(layer_num, module_name='fc1', MAX=500):

    # Set the seed. You don't need indices if data is loaded in same order every time.
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = CLIPModel.from_pretrained("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
    processor = CLIPProcessor.from_pretrained("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", do_rescale=False) # Make sure the do_rescale is false for pytorch datasets

    imagenet_data = datasets.ImageFolder(imagenet_path, transform=data_transforms)

    save_path = f'/network/scratch/s/sonia.joseph/clip_mechinterp/tinyclip/mini_dataset/'
    file_name = f'mlp_{module_name}_{layer_num}.npz'

    # Make directory if doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Example usage
        
    module = getattr(model.vision_model.encoder.layers[layer_num].mlp, module_name)

    hook_handle = register_hook(module)

    master_layer_activations = []
    count = 0
    for batch_idx, (total_images, total_labels) in tqdm(enumerate(data_loader), total=MAX):
        detailed_activations = process_images(model, processor, total_images, total_labels, batch_idx=batch_idx)
        master_layer_activations.append(detailed_activations)
        count += 1
        if count >= MAX:
            break

    # Remove the hook when done
    hook_handle.remove()

    flattened_activations = [item for sublist in master_layer_activations for item in sublist]
    df_activations = pd.DataFrame(flattened_activations)

    parquet_file_path = os.path.join(save_path, file_name)
    df_activations.to_parquet(parquet_file_path, index=False)

    print(f"Saved {layer_num} as parquet.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process and save model activations")
    parser.add_argument("--layer_num", type=int, required=True, help="Model layer number to hook")
    parser.add_argument("--module_name", type=str, default='fc1', required=False, help="Module name to hook")
    parser.add_argument("--MAX", type=int, default=500, required=False, help="Number of images to process")

    args = parser.parse_args()
    main(args.layer_num, args.module_name, args.MAX)
