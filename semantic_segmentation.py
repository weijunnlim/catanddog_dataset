from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np

# hugging face link: https://huggingface.co/docs/transformers/en/model_doc/segformer
image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

image_path = "/home/dxd_wj/catanddog_dataset/images/image2.jfif"
image = Image.open(image_path).convert("RGB")
inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

predictions = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

original_size = image.size
predictions_pil = Image.fromarray(predictions.astype(np.uint8)) 
upsampled_predictions_pil = predictions_pil.resize(original_size, Image.NEAREST) 

# Convert the upsampled predictions to numpy array
upsampled_predictions_np = np.array(upsampled_predictions_pil)

# Create a colormap for visualization
cmap = plt.get_cmap('tab20b', np.max(upsampled_predictions_np) + 1)  # Adjust colormap as needed

# Plot the results
plt.imshow(upsampled_predictions_np, cmap=cmap)
plt.colorbar()
plt.title('Segmentation Output')
plt.axis('off')
plt.show()

# Save the segmented image
segmentation_image = Image.fromarray(upsampled_predictions_np.astype(np.uint8))
segmentation_image.save("segmentation_output.png")