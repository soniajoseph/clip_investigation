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

import argparse
import pandas as pd

from transformers import CLIPProcessor, CLIPModel

from PIL import Image
import torchvision.transforms as transforms
import torch
import copy
from scipy.special import logsumexp


from collections import defaultdict



from PIL import Image
import torchvision.transforms as transforms
import torch
import copy
from scipy.special import logsumexp



cup_classes = [
    "measuring_cup",
    "coffee_mug",
    "water_jug",
    "whiskey_jug",
    "beer_bottle",
    "pill_bottle",
    "pop_bottle",
    "water_bottle",
    "wine_bottle",
    "washbasin",
    "beaker",
    "vase",
    "cauldron",
    "coffeepot",
    "teapot",
    "barrel",
    "bathtub",
    "bucket",
    "ladle",
    "mortar",
    "pitcher",
    "tub",
    "mixing_bowl",
    "soup_bowl",
    "Petri dish",
    "milk_can",
    "beer_glass",
    "goblet",
    "cocktail_shaker",
    "saltshaker",
    "pot",
    "thimble",
    "hot_pot",
    "trifle",
    "consomme",
    "espresso",
    "red_wine",
    "cup",
    "eggnog"
]
dog_classes = [
    "chihuahua",
    "japanese_spaniel",
    "maltese_dog",
    "pekinese",
    "shih-tzu",
    "blenheim_spaniel",
    "papillon",
    "toy_terrier",
    "rhodesian_ridgeback",
    "afghan_hound",
    "basset",
    "beagle",
    "bloodhound",
    "bluetick",
    "black-and-tan_coonhound",
    "walker_hound",
    "english_foxhound",
    "redbone",
    "borzoi",
    "irish_wolfhound",
    "italian_greyhound",
    "whippet",
    "ibizan_hound",
    "norwegian_elkhound",
    "otterhound",
    "saluki",
    "scottish_deerhound",
    "weimaraner",
    "staffordshire_bullterrier",
    "american_staffordshire_terrier",
    "bedlington_terrier",
    "border_terrier",
    "kerry_blue_terrier",
    "irish_terrier",
    "norfolk_terrier",
    "norwich_terrier",
    "yorkshire_terrier",
    "wire-haired_fox_terrier",
    "lakeland_terrier",
    "sealyham_terrier",
    "airedale",
    "cairn",
    "australian_terrier",
    "dandie_dinmont",
    "boston_bull",
    "miniature_schnauzer",
    "giant_schnauzer",
    "standard_schnauzer",
    "scotch_terrier",
    "tibetan_terrier",
    "silky_terrier",
    "soft-coated_wheaten_terrier",
    "west_highland_white_terrier",
    "lhasa",
    "flat-coated_retriever",
    "curly-coated_retriever",
    "golden_retriever",
    "labrador_retriever",
    "chesapeake_bay_retriever",
    "german_short-haired_pointer",
    "vizsla",
    "english_setter",
    "irish_setter",
    "gordon_setter",
    "brittany_spaniel",
    "clumber",
    "english_springer",
    "welsh_springer_spaniel",
    "cocker_spaniel",
    "sussex_spaniel",
    "irish_water_spaniel",
    "kuvasz",
    "schipperke",
    "groenendael",
    "malinois",
    "briard",
    "kelpie",
    "komondor",
    "old_english_sheepdog",
    "shetland_sheepdog",
    "collie",
    "border_collie",
    "bouvier_des_flandres",
    "rottweiler",
    "german_shepherd",
    "doberman",
    "miniature_pinscher",
    "greater_swiss_mountain_dog",
    "bernese_mountain_dog",
    "appenzeller",
    "entlebucher",
    "boxer",
    "bull_mastiff",
    "tibetan_mastiff",
    "french_bulldog",
    "great_dane",
    "saint_bernard",
    "eskimo_dog",
    "malamute",
    "siberian_husky"
]


def get_imagenet_classes():

    def get_class_name(label):
    # Assuming the label maps to a list of class names
        return num_to_word_dict.get(str(label), ["Unknown label"])[1]

    json_file_path = '/home/mila/s/sonia.joseph/ViT-Planetarium/my_draft/test_nbs/imagenet_class_index.json'
    word_to_num_dict = {}

    # Load the JSON file into a Python dictionary
    with open(json_file_path, 'r') as file:
        num_to_word_dict = json.load(file)

    for num, words in num_to_word_dict.items():
        for word in words:  # Assuming each entry in num_to_word_dict is a list of words
            word_to_num_dict[word] = num
 
    imagenet_class_nums = np.arange(0, 1000, 1)
    imagenet_class_names = ["{}".format(get_class_name(i)) for i in imagenet_class_nums]
    return imagenet_class_names

imagenet_class_names = get_imagenet_classes()


def load_cached_act(layer_num, save_path = '/network/scratch/s/sonia.joseph/clip_mechinterp/tinyclip/mini_dataset/', layer_type='fc1'):
    """
    Load cached activations and calculate per-neuron z-scores.
    """

    file_name = f'mlp_{layer_type}_{layer_num}.npz'
    loaded = pd.read_parquet(os.path.join(save_path, file_name))
    return loaded



def logit_metric(logits, cup_indices):
    # Convert cup_indices to a tensor if it's a list
    cup_indices_tensor = torch.tensor(cup_indices, dtype=torch.long)

    # Create a boolean mask for all logits: False for cup indices, True for others

    mask = torch.ones(logits.shape[1], dtype=torch.bool)  # All True initially
    mask[cup_indices_tensor] = False  # Set False for cup indices

    # Extract cup logits
    cup_logit = logits[:, cup_indices_tensor]

    # Extract non-cup logits using the inverted mask
    non_cup_logit = logits[:, mask]

    diff = logsumexp(cup_logit, axis=1).mean() - logsumexp(non_cup_logit, axis=1).mean()

    return diff

def create_custom_hook(neuron_idx, new_value):
    # This is the actual hook function
    def custom_forward_hook(module, input, output):
        # Modify the output for the specified neuron
        output[:, :, neuron_idx] = new_value
        return output
    return custom_forward_hook

# Path to your PNG image

# Load the image


def ablate_and_get_logit_diff(image_tensor, idx, neuron_idx, layer_num, resampling_activations, model, processor, vanilla_logits, indices=None):


    inputs = processor(text=imagenet_class_names, images=image_tensor, return_tensors="pt", padding=True)

    new_value = resampling_activations.iloc[neuron_idx]  # example new value
    custom_hook_function = create_custom_hook(neuron_idx, new_value)
    ablated_model = copy.deepcopy(model)
    hook = ablated_model.vision_model.encoder.layers[layer_num].mlp.fc1.register_forward_hook(custom_hook_function)

    ablated_outputs = ablated_model(**inputs)
    logits_per_image = ablated_outputs.logits_per_image
    ablated_logits = logits_per_image.detach().numpy()

    hook.remove()

    vanilla_log_diff = logit_metric(vanilla_logits, indices)
    # print("Vanilla logit diff", vanilla_log_diff)

    abl_log_diff = logit_metric(ablated_logits, indices)
    # print("Ablated logit diff", abl_log_diff)

    del ablated_model

    return vanilla_log_diff, abl_log_diff


def percent_change(old_value, new_value):
    return (new_value - old_value) / np.abs(old_value) * 100


def transform_images(img_directory, image_path):
    image = Image.open(os.path.join(img_directory, image_path)).convert('RGB')

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size expected by the CLIP model
        transforms.ToTensor(),          # Convert to PyTorch Tensor
        # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])  # Normalize (specific to CLIP)
    ])

    # Apply the transformation
    image_tensor = transform(image)

    # Add a batch dimension
    image_tensor = image_tensor.unsqueeze(0) 
    return image_tensor

def calculate_logit_diff_dictionary_per_class(image_list, class_type, indices, layer_num, layer_type):

    if layer_type == 'fc1':
        max_neuron = 1024
    elif layer_type == 'fc2':
        max_neuron = 256
    
    neuron_dict = defaultdict(list)
    for idx, image_tensor in tqdm(enumerate(image_list)):
        print("On image", idx)
        inputs = processor(text=imagenet_class_names, images=image_tensor, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        vanilla_logits = logits_per_image.detach().numpy()
        for neuron_idx in tqdm(range(max_neuron)):
            vanilla, abl = ablate_and_get_logit_diff(image_tensor, idx, neuron_idx=neuron_idx, layer_num=layer_num, resampling_activations=avg_dog_activation,
                                                    model=model, processor=processor, vanilla_logits=vanilla_logits, indices=indices)
            neuron_dict[neuron_idx].append(percent_change(vanilla, abl))

    # Save dictionary
    save_dir = '/network/scratch/s/sonia.joseph/clip_mechinterp/tinyclip/logit_differences/cup_logits'
    np.save(os.path.join(save_dir, f'neuron_dict_{layer_num}_{class_type}_{layer_type}.npy'), neuron_dict)
    print("Done")

if __name__ == '__main__':

    # argparse

    parser = argparse.ArgumentParser(description="Process and save model activations")
    parser.add_argument("--layer_num", type=int, required=True, help="Model layer number to hook")
    parser.add_argument("--layer_type", type=str, required=True, help="MLP fc1 or fc2")

    args = parser.parse_args()

    layer_num = args.layer_num
    layer_type = args.layer_type

    model = CLIPModel.from_pretrained("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
    processor = CLIPProcessor.from_pretrained("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", do_rescale=False) # Make sure the do_rescale is false for pytorch datasets

    # Load activations and get resampling activations
    loaded = load_cached_act(layer_num, layer_type=layer_type)
    print("Loaded activations.")
    avg_dog_activation = loaded[loaded['class_name'].isin(dog_classes)].groupby('neuron_idx')['activation_value'].mean()

    cup_dir = '/home/mila/s/sonia.joseph/CLIP_mechinterp/sample_images/cup_images'
    new_cup_images = ['beer.png','many_cups.png','blue_cup.png','two_cups.png','tub.png']

    random_dir = '/home/mila/s/sonia.joseph/CLIP_mechinterp/sample_images/random_images'
    random_images = ['banana.png', 'brocolli.png', 'cat.png', 'dog.png', 'hen.png', 'salamander.png']

    dog_dir = '/home/mila/s/sonia.joseph/CLIP_mechinterp/sample_images/dog_images'
    dog_images = ['dalmation.png', 'husky.png', 'pomeranian.png', 'malamute.png']

    # Apply the transformation
    new_cup_images = [transform_images(cup_dir, image_path) for image_path in new_cup_images]
    random_images = [transform_images(random_dir, image_path) for image_path in random_images]
    dog_images = [transform_images(dog_dir, image_path) for image_path in dog_images]

        
    cup_indices = [imagenet_class_names.index(cup_name) for cup_name in cup_classes if cup_name in imagenet_class_names]
    dog_indices = [imagenet_class_names.index(dog_name) for dog_name in dog_classes if dog_name in imagenet_class_names]

    calculate_logit_diff_dictionary_per_class(new_cup_images, 'cups', cup_indices, layer_num, layer_type)
    # calculate_logit_diff_dictionary_per_class(random_images, 'random', cup_indices, layer_num, layer_type)
    # calculate_logit_diff_dictionary_per_class(dog_images, 'dogs', dog_indices, layer_num, layer_type)

    




