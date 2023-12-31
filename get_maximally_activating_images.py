import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import logging
import json
import random

import numpy as np
import json
from tqdm.auto import tqdm

import pandas as pd
import os

import argparse

import matplotlib.pyplot as plt


# Load imagenet

def load_imagenet(imagenet_path='/network/datasets/imagenet.var/imagenet_torchvision/val/'):

    # Path to your imagenet_class.json file

    # Get class names
    # imagenet_class_nums = np.arange(0, 1000, 1)
    # imagenet_class_names = ["{}".format(get_class_name(i)) for i in imagenet_class_nums]

    # Set the seed. You don't need indices if data is loaded in same order every time.
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # # Load the ImageNet dataset
    imagenet_data = datasets.ImageFolder(imagenet_path, transform=data_transforms)
    return imagenet_data


def load_cached_act(layer_num, save_path = '/network/scratch/s/sonia.joseph/clip_mechinterp/tinyclip/mini_dataset/'):
    """
    Load cached activations and calculate per-neuron z-scores.
    """

    file_name = f'mlp_fc1_{layer_num}.npz'
    loaded = pd.read_parquet(os.path.join(save_path, file_name))

    # Calculate standard deviation
    # Calculate mean and standard deviation for 'activation_value' grouped by 'neuron_idx'
    grouped = loaded.groupby('neuron_idx')['activation_value']
    mean_per_neuron = grouped.transform('mean')
    std_dev_per_neuron = grouped.transform('std')

    # Calculate the z-score (number of standard deviations from the mean)
    loaded['activation_value_sds'] = (loaded['activation_value'] - mean_per_neuron) / std_dev_per_neuron

    # Replace NaN and infinite values (which can occur if std_dev is zero) with zero
    loaded['activation_value_sds'].replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    # Sort by z-score
    # sorted_df = loaded.sort_values(by=['activation_value_sds'], ascending=False)
    return loaded


# def get_max_and_min_images(df, layer_num, num_return=10, save_dir = '/home/mila/s/sonia.joseph/CLIP_mechinterp/results/max_min_images'):

#     # Ensure 'neuron_idx' is not part of the index
#     top_patches_per_neuron = df.reset_index(drop=True)

#     # Group by 'neuron_idx', then sort within groups by 'activation_value'
#     grouped = top_patches_per_neuron.groupby('neuron_idx', group_keys=False).apply(lambda x: x.sort_values('activation_value_sds', ascending=False))

#     # Get top 5 entries for each neuron
#     top_per_neuron = grouped.groupby('neuron_idx').head(num_return)
#     bottom_per_neuron = grouped.groupby('neuron_idx').tail(num_return)

#     column_names = ['batch_idx', 'neuron_idx', 'patch_idx', 'class_name', 'predicted', 'activation_value', 'activation_value_sds']


#     # Selecting the relevant columns
#     top_per_neuron = top_per_neuron[column_names]
#     top_per_neuron.to_csv(os.path.join(save_dir, f'top_per_neuron_layer_{layer_num}.csv'), index=False)

#       # Selecting the relevant columns
#     bottom_per_neuron = bottom_per_neuron[column_names]
#     bottom_per_neuron.to_csv(os.path.join(save_dir, f'bottom_per_neuron_layer_{layer_num}.csv'), index=False)

#     return top_per_neuron, bottom_per_neuron



def plot_image_patch_heatmap(activation_values_array, image, specific_neuron_idx, image_size=224, pixel_num=14, class_name=None):

    activation_values_array = activation_values_array.reshape(pixel_num, pixel_num)

    # Create a heatmap overlay
    heatmap = np.zeros((image_size, image_size))
    patch_size = image_size // pixel_num

    for i in range(pixel_num):
        for j in range(pixel_num):
            heatmap[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = activation_values_array[i, j]

    # Plotting the image with the heatmap overlay
    fig, ax = plt.subplots()
    ax.imshow(image.permute(1,2,0))
    ax.imshow(heatmap, cmap='viridis', alpha=0.54)  # Overlaying the heatmap

    # Removing axes
    ax.axis('off')

    min_activation = activation_values_array.min()
    max_activation = activation_values_array.max()

    # Adding colorbar for the heatmap
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min_activation, vmax=max_activation)), ax=ax, orientation='vertical')
    plt.title("{}".format(class_name))

    # Return plot
    return plt

def get_image_and_activations_by_id(loaded, specific_neuron_idx, specific_image_idx, imagenet_data, indices_path='/home/mila/s/sonia.joseph/CLIP_mechinterp/imagenet_sample_indices.npy'):
    '''
    Enter loaded df.
    '''
    random_indices = np.load(indices_path)
    image, label = load_specific_image(imagenet_data, random_indices, specific_image_idx)
    filtered_df = loaded[(loaded['batch_idx'] == specific_image_idx) & 
                                (loaded['neuron_idx'] == specific_neuron_idx)]
    activation_values = filtered_df['activation_value_sds']
    activation_values_array = activation_values.to_numpy()[1:]
    return image, activation_values_array


# Function to load a specific image
def load_specific_image(dataset, indices, order):
    specific_index = indices[order]  
    image, label = dataset[specific_index]
    return image, label


def visualize_top_n_results(loaded_df, sorted_df, layer_num, specific_neuron_idx, return_n=20, save_dir=None,
                            imagenet_data=None):

    # layer_dir = os.path.join(save_dir, f'layer_{layer_num}')
    # # Create directory if doesn't exist
    # if not os.path.exists(layer_dir):
    #     os.makedirs(layer_dir)

    unique_top_entries = sorted_df[sorted_df['neuron_idx'] == specific_neuron_idx].drop_duplicates(subset='class_name', keep='first').head(return_n)
    unique_bottom_entries = sorted_df[sorted_df['neuron_idx'] == specific_neuron_idx].drop_duplicates(subset='class_name', keep='last').tail(return_n)

    # Extracting class names and activation values
    # unique_top_class_names = unique_top_entries['class_name'].tolist()
    # unique_top_activations = unique_top_entries['activation_value'].tolist()


    unique_top_batch_idx = unique_top_entries['batch_idx'].tolist()
    unique_top_class_names = unique_top_entries['class_name'].tolist()

    unique_bottom_batch_idx = unique_bottom_entries['batch_idx'].tolist()
    unique_bottom_class_names = unique_bottom_entries['class_name'].tolist()

    # Save top_unique_entries as df
    unique_top_entries.to_csv(os.path.join(layer_dir, f'max_imgs_neuron_{specific_neuron_idx}.csv'), index=False)
    unique_bottom_entries.to_csv(os.path.join(layer_dir, f'min_imgs_neuron_{specific_neuron_idx}.csv'), index=False)

    # # Lists are ready to use
    # print("Top 10 Unique Class Names:", unique_top_class_names)
    # print("Corresponding Activations:", unique_top_activations)
    # print("Corresponding Batch Indices:", unique_top_batch_idx)

    for i, (batch_idx, class_name) in enumerate(zip(unique_top_batch_idx, unique_top_class_names)):
        image, activation_values_array = get_image_and_activations_by_id(loaded_df, specific_neuron_idx, batch_idx, imagenet_data)
        plot = plot_image_patch_heatmap(activation_values_array, image, specific_neuron_idx, image_size=224, class_name=class_name)

        # Save plot, no axes
        plot.savefig(os.path.join(layer_dir, f'neuron_{specific_neuron_idx}_max_{i}.png'), bbox_inches='tight')

        # Save as svg
        plot.savefig(os.path.join(layer_dir, f'neuron_{specific_neuron_idx}_max_{i}.svg'), bbox_inches='tight')
        plot.close()

    
    for i, (batch_idx, class_name) in enumerate(zip(unique_bottom_batch_idx, unique_bottom_class_names)):
        image, activation_values_array = get_image_and_activations_by_id(loaded_df, specific_neuron_idx, batch_idx, imagenet_data)
        plot = plot_image_patch_heatmap(activation_values_array, image, specific_neuron_idx, image_size=224, class_name=class_name)
        plot.savefig(os.path.join(layer_dir, f'neuron_{specific_neuron_idx}_min_{i}.png'), bbox_inches='tight')

        # Save as svg
        plot.savefig(os.path.join(layer_dir, f'neuron_{specific_neuron_idx}_min_{i}.svg'), bbox_inches='tight')
        plot.close()

        # print("Class Name:", unique_top_class_names[i])
        # print("Activation Value:", unique_top_activations[i])
        # print("Batch Index:", unique_top_batch_idx[i])
        # print("")

# Write main 
if __name__ == '__main__':

    # Enter layer number
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_num', type=int, help='Layer number')
    parser.add_argument('--num_neurons', default=1024, type=int, help='Number of neurons to visualize')

    args = parser.parse_args()
    layer_num = args.layer_num
    print("Running layer number:", layer_num)

    # Load imagenet
    imagenet_data = load_imagenet()

    layer_dir = os.path.join("/network/scratch/s/sonia.joseph/clip_mechinterp/tinyclip/max_min_imgs", f'layer_{layer_num}')
    # Make directory if doesn't exist
    if not os.path.exists(layer_dir):
        os.makedirs(layer_dir)

    # Load cached activations
    loaded_df = load_cached_act(layer_num)
    sorted_df = loaded_df.sort_values(by=['activation_value_sds'], ascending=False)

    print("Loaded cached activations and sorted by z-score.")
    print("Saving to csv...")


    # Save sorted to csv, but only head and tail
    sorted_df.head(1000).to_csv(os.path.join(layer_dir, 'sorted_head.csv'), index=False)
    sorted_df.tail(1000).to_csv(os.path.join(layer_dir, 'sorted_tail.csv'), index=False)

    print("Visualizing max/min results for each neurons")

    for neuron_idx in tqdm(range(args.num_neurons)):
        # if file exists, skip
        # if os.path.exists(os.path.join(layer_dir, f'neuron_{neuron_idx}_max_0.png')):
        #     print("skipping, already exists.")
        #     continue

        visualize_top_n_results(loaded_df, sorted_df, layer_num, specific_neuron_idx=neuron_idx, return_n=10, save_dir = layer_dir, imagenet_data=imagenet_data)

    print("Done.")
