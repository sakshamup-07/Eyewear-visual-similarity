import os
import glob
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

LABELS = [
    "Aviator eyeglasses",
    "Round frame eyeglasses",
    "Square frame eyeglasses",
    "Wayfarer eyeglasses",
    "Rimless eyeglasses",
    "Transparent frame eyeglasses"
]

label_to_style = {
    "Aviator eyeglasses": "Aviator",
    "Round frame eyeglasses": "Round",
    "Square frame eyeglasses": "Square",
    "Wayfarer eyeglasses": "Wayfarer",
    "Rimless eyeglasses": "Rimless",
    "Transparent frame eyeglasses": "Transparent"
}

df = pd.read_csv("dataset/metadata.csv")

new_styles = []

for idx, row in df.iterrows():
    image_path = row["image_path"]

    base = os.path.splitext(image_path)[0]     
    matches = glob.glob(base + ".*")           # finds .jpg, .png, .jpeg

    if len(matches) == 0:
        raise FileNotFoundError(f"Image not found for: {base}")

    image = Image.open(matches[0]).convert("RGB")

    inputs = processor(text=LABELS, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1).cpu().numpy()[0]

    best_label = LABELS[probs.argmax()]
    predicted_style = label_to_style[best_label]
    new_styles.append(predicted_style)

    print(f"{matches[0]} → {predicted_style}")

df["shape"] = new_styles
df.to_csv("dataset/metadata.csv", index=False)

print("\n✅ Metadata updated with AI-generated styles.")
