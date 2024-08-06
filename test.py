import torch
from torchvision.models import mobilenet_v2
from PIL import Image
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_image(model, image):
    #set model to evaluate mode
    model.eval() 
    with torch.no_grad():  # no grad to save computation
        output = model(image)
        _, predicted = torch.max(output, 1) #to determine which class has higher score
        prediction = predicted.item() # 0 or 1
    
    class_labels = ['Cat', 'Dog']
    return class_labels[prediction]

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load the model
model_path = '/home/dxd_wj/catanddog_dataset/model.pth'
#model = mobilenet_v2()  # Initialize the model architecture
model = torch.load(model_path)  
#model.load_state_dict(input_params)
model.to(device)  

 #its time to test
image_path = '/home/dxd_wj/catanddog_dataset/images/image6.jpg'
image = Image.open(image_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0)
image= image.to(device)
prediction = predict_image(model, image)
print(f'The image is predicted as: {prediction}')

