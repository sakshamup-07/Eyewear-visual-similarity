import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

labels = [
    "Aviator eyeglasses",
    "Round frame eyeglasses",
    "Square frame eyeglasses",
    "Wayfarer eyeglasses",
    "Rimless eyeglasses",
    "Transparent frame eyeglasses"
]

def predict_attributes(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    scores = probs[0].cpu().numpy()
    results = list(zip(labels, scores))
    results.sort(key=lambda x: x[1], reverse=True)

    return results

if __name__ == "__main__":
    test_image = "dataset/images/000001.jpeg"
    preds = predict_attributes(test_image)

    for label, score in preds:
        print(f"{label}: {score:.3f}")
