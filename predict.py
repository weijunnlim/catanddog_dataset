# import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import CustomImageDataset


csv_dataset = pd.read_csv('C:/Users/tp-limwj/Downloads/cat_dog.csv')
root_dataset = 'C:/Users/tp-limwj/Downloads/cat_dog'

train_data, intermediate_data = train_test_split(csv_dataset, test_size=0.2, random_state=42) #80% train 20% inter
test_data, val_data = train_test_split(intermediate_data, test_size = 0.5, random_state = 42)

train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

train_dataset = CustomImageDataset(csv_file='train_data.csv', root_dir=root_dir, transform=transform)
val_dataset = CustomImageDataset(csv_file='val_data.csv', root_dir=root_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Visualize images
from torchvision.transforms.functional import to_pil_image

def show_batch(dl):
    for images, labels in dl:
        fig, axes = plt.subplots(1, len(images), figsize=(12, 4))
        for img, ax in zip(images, axes):
            ax.imshow(to_pil_image(img))  # Convert tensor back to PIL image for displaying
            ax.axis('off')
        plt.show()
        break  # Remove this break to display more batches

# Display a batch of training images
print("Training Images:")
show_batch(train_loader)

# Display a batch of validation images
print("Validation Images:")
show_batch(val_loader)