import os
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pinecone import Pinecone

# ------------------ CONFIG ------------------
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = st.secrets["PINECONE_INDEX"]
CSV_PATH = "dataset/metadata.csv"

# ------------------ INIT ------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ------------------ LOAD METADATA ------------------
df = pd.read_csv(CSV_PATH)

vectors = []

for _, row in df.iterrows():
    image_path = row["image_path"]
    img_id = str(row["id"])

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

    vector = embedding[0].cpu().numpy().tolist()

    metadata = {
        "brand": row["brand"],
        "price": int(row["price"]),
        "material": row["material"],
        "shape": row["shape"]
    }

    vectors.append((img_id, vector, metadata))
    print(f"Processed image {img_id}")

# ------------------ UPSERT TO PINECONE ------------------
index.upsert(vectors)
print("All images successfully ingested into Pinecone.")
