import torch
from torchvision.models import mobilenet_v2
from PIL import Image
from torchvision import transforms

def predict_image(model, image):
    #set model to evaluate mode
    model.eval() 
    with torch.no_grad():  # no grad to save computation
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()
    
    class_labels = ['Cat', 'Dog', 'None of the Above']
    return class_labels[prediction]

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

model = mobilenet_v2()
model_path = '/home/dxd_wj/catanddog_dataset/model.pth'
state_dict = torch.load(model_path, weights_only = True)
model.load_state_dict(state_dict)

 #its time to test
image_path = '/home/dxd_wj/catanddog_dataset/images/image1.jfif'
image = Image.open(image_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0)
prediction = predict_image(model, image)
print(f'The image is predicted as: {prediction}')

