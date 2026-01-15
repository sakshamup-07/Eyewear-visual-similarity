from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load a test image (put one glasses image in same folder and name it test.jpg)
image = Image.open("test.jpg").convert("RGB")

# Preprocess and get embedding
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    image_features = model.get_image_features(**inputs)

# Normalize embedding
image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

print("Embedding shape:", image_features.shape)
print("First 10 values of vector:", image_features[0][:10])
