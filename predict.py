# import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.optim as optim
import torch
import torch.nn as nn
from custom_dataset import CustomImageDataset
from PIL import Image

def main():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    csv_dataset = pd.read_csv('/home/dxd_wj/catanddog_dataset/cat_dog.csv') #to change
    root_dataset = '/home/dxd_wj/catanddog_dataset/cat_dog' #to change

    train_data, intermediate_data = train_test_split(csv_dataset, test_size=0.2, random_state=42) # 80% train, 20% intermediate
    test_data, val_data = train_test_split(intermediate_data, test_size=0.5, random_state=42) # 80% train, 10% val, 10% test

    train_data.to_csv('train_data.csv', index=False)
    val_data.to_csv('val_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)

    train_dataset = CustomImageDataset(csv_file='train_data.csv', root_dir=root_dataset, transform=transform)
    val_dataset = CustomImageDataset(csv_file='val_data.csv', root_dir=root_dataset, transform=transform)
    test_dataset = CustomImageDataset(csv_file='test_data.csv', root_dir=root_dataset, transform=transform)

    # small_train_dataset = torch.utils.data.Subset(train_dataset, range(2000))
    # small_train_loader = DataLoader(small_train_dataset, batch_size=8, shuffle=True)

    # small_val_dataset = torch.utils.data.Subset(val_dataset, range(200))
    # small_val_loader = DataLoader(small_val_dataset, batch_size=8, shuffle=True)

    # small_test_dataset = torch.utils.data.Subset(test_dataset, range(200))
    # small_test_loader = DataLoader(small_test_dataset, batch_size=8, shuffle=True)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # load pre trained model    
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    #print(model)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 3)  # last layer depends on how many classes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) #using Adam optimizer

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')
    torch.save(model, 'model.pth')
    print("Entire model saved to 'model.pth'")

    #its time to test
#     image_path = '/Users/limweijun/Downloads/image_1.jpeg'
#     image = Image.open(image_path).convert("RGB")
#     image = transform(image)
#     image = image.unsqueeze(0)
#     prediction = predict_image(model, image)
#     print(f'The image is predicted as: {prediction}')

# def predict_image(model, image):
#     model.eval() 
#     with torch.no_grad(): 
#         output = model(image)
#         _, predicted = torch.max(output, 1)
#         prediction = predicted.item()
    
#     class_labels = ['Cat', 'Dog', 'None of the Above']
#     return class_labels[prediction]

if __name__ == '__main__':
    main()