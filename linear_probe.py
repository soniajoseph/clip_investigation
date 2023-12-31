import os
import torch
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel
from torchvision import datasets, transforms

from torch.utils.data import DataLoader, SubsetRandomSampler


#  LOAD IMAGENET
def load_imagenet(subset_size=10000, train_split=0.8):

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


    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    imagenet_data = datasets.ImageFolder(imagenet_path, transform=data_transforms)

   # Splitting into train and test subsets
    total_size = len(imagenet_data)
    indices = torch.randperm(total_size).tolist()
    split = int(np.floor(train_split * subset_size))

    train_indices, test_indices = indices[:split], indices[split:subset_size]

    # Creating data samplers
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Creating data loaders
    train_loader = DataLoader(imagenet_data, sampler=train_sampler, batch_size=100)
    test_loader = DataLoader(imagenet_data, sampler=test_sampler, batch_size=100)

    return train_loader, test_loader, imagenet_class_names


def register_hook(module):
    def hook(module, input, output):
        # print(output[0].shape)
        activations_list.append(output[0].detach())
    return module.register_forward_hook(hook)

def get_features(dataloader, imagenet_class_names): # layer to train on
    all_features = []
    all_labels = []

    activations_list.clear()
    
    max = 10
    count = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):

            inputs = processor(text=imagenet_class_names, images=images, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            total_labels.append(labels)

            # class_name = get_class_name(labels.item())
            # logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            # probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
            # predicted_class_name = imagenet_class_names[probs.argmax(dim=1)]

            count += 1
            if count > max:
                break
    return activations_list, torch.cat(total_labels).cpu().numpy()




model = CLIPModel.from_pretrained("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
processor = CLIPProcessor.from_pretrained("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", do_rescale=False) # Make sure the do_rescale is false for pytorch datasets

train_loader, test_loader, imagenet_class_names = load_imagenet(subset_size=10000) # get 10k samples

layer_num = 7
module_name = 'fc2'
module = getattr(model.vision_model.encoder.layers[layer_num].mlp, module_name)

register_hook(module)

activations_list = []
total_labels = []

# Calculate the image features
print("Collecting intermediate actiavitions")
train_features, train_labels = get_features(train_loader, imagenet_class_names)
test_features, test_labels = get_features(test_loader, imagenet_class_names)

# Save the features
print("Saving features")
save_dir = f'/network/scratch/s/sonia.joseph/clip_mechinterp/tinyclip/intermediate_acts'
np.save(os.path.join(save_dir, f"train_features_layer_{layer_num}.npy"), train_features,)
np.save(os.path.join(save_dir, f"train_labels_layer_{layer_num}.npy"), train_labels)
np.save(os.path.join(save_dir, f"test_features_layer_{layer_num}.npy"), test_features)
np.save(os.path.join(save_dir, f"test_labels_layer_{layer_num}.npy"), test_labels)


print("Fitting logistic regression model")

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

# Assuming train_features and train_labels are PyTorch tensors
input_dim = train_features.shape[-1]
output_dim = 1000 # or the number of classes in your case

model = LogisticRegressionModel(input_dim, output_dim)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):  # number of epochs
    model.train()
    optimizer.zero_grad()

    # Get intermediate activations as train_features

    outputs = model(train_features.to(device))
    loss = criterion(outputs, train_labels.to(device))
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    outputs = model(test_features.to(device))
    _, predicted = torch.max(outputs.data, 1)
    total += test_labels.size(0)
    correct += (predicted == test_labels.to(device)).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.3f}%')