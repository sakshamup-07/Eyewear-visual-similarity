import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pinecone import Pinecone

PINECONE_API_KEY = "pcsk_6E6LwA_EZP8x4Zo4yt4QJGoeE6jt6GE6Zq7sRVN9Zi1Q7zMiiLMzAVAq5HdDBdt5BD5dzW"
INDEX_NAME = "lenskart-eyewear"

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def search_similar(image_path, top_k=5, filters=None):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

    vector = embedding[0].cpu().numpy().tolist()

    results = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=filters
    )

    return results

if __name__ == "__main__":
    query_image = "dataset/images/000001.jpeg"  # test image
    results = search_similar(query_image)

    for match in results["matches"]:
        print("ID:", match["id"])
        print("Similarity Score:", match["score"])
        print("Metadata:", match["metadata"])
        print("-" * 40)
