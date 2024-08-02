from transformers import AutoImageProcessor, AutoModelForImageClassification
import PIL
import requests

image_path = "/Users/limweijun/Downloads/image_2.jpeg"
image = PIL.Image.open(image_path)
#image = PIL.Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("wesleyacheng/dog-breeds-multiclass-image-classification-with-vit")
model = AutoModelForImageClassification.from_pretrained("wesleyacheng/dog-breeds-multiclass-image-classification-with-vit")

inputs = image_processor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits

# model predicts one of the 120 Stanford dog breeds classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])