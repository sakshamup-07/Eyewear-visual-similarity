import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from pinecone import Pinecone
import numpy as np
import os
import json
import math
import time
import logging
import cv2

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger()

#  CONFIG 
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = st.secrets["PINECONE_INDEX"]
FEEDBACK_FILE = "style_feedback.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

#  LOAD MODEL 
@st.cache_resource
def load_model():
    return CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)

@st.cache_resource
def load_processor():
    return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model = load_model()
processor = load_processor()

#  LABELS FOR ATTRIBUTE TAGGING 
labels = [
    "Aviator eyeglasses",
    "Round frame eyeglasses",
    "Square frame eyeglasses",
    "Wayfarer eyeglasses",
    "Rimless eyeglasses",
    "Transparent frame eyeglasses"
]

#  FEEDBACK UTILS 
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump({}, f)

def load_feedback():
    with open(FEEDBACK_FILE, "r") as f:
        return json.load(f)

def save_feedback(data):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)
       
       #  COLOR EXTRACTION 
def detect_frame_color(image):
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    pixels = hsv.reshape((-1, 3))
    pixels = np.float32(pixels)

    K = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    dominant = centers[np.argmax(np.bincount(labels.flatten()))]
    hue, sat, val = dominant

    if val < 50:
        return "Black"
    elif val > 200 and sat < 30:
        return "White"
    elif 35 <= hue <= 85:
        return "Green"
    elif 90 <= hue <= 130:
        return "Blue"
    elif 10 <= hue <= 25:
        return "Yellow"
    elif hue <= 10 or hue >= 170:
        return "Red"
    else:
        return "Brown / Mixed"




#  EMBEDDING FUNCTIONS
def get_image_embedding(image):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb[0].cpu().numpy()

def get_text_embedding(text):
    inputs = processor(text=text, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb[0].cpu().numpy()

def predict_attributes(image):
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
    scores = probs[0].cpu().numpy()
    return list(zip(labels, scores))

#  STREAMLIT UI 
st.set_page_config(page_title="Lenskart Visual Search", layout="wide")
st.title("👓 Lenskart Visual Similarity Search (AI Powered)")

uploaded_file = st.file_uploader("Upload an image of glasses", type=["jpg", "jpeg", "png"])
text_query = st.text_input("Optional: Refine with text (e.g. 'round metal frame', 'transparent rim')")

if uploaded_file is None:
    st.info("Please upload an image to start the search.")
    st.stop()

# ---- Safe Image Load ----
try:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Query Image", width=300)
    frame_color = detect_frame_color(image)



except Exception as e:
    logger.error(f"Corrupt image upload: {e}")
    st.error("Invalid or corrupted image file.")
    st.stop()

# ---- Zero-shot Attribute Detection 
st.subheader("🧠 Detected Frame Attributes (Zero-Shot CLIP)")
attrs = predict_attributes(image)
for label, score in attrs:
    st.write(f"{label} : {score:.2f}")

predicted_label = max(attrs, key=lambda x: x[1])[0]
predicted_shape = predicted_label.replace(" eyeglasses", "").replace(" frame", "").strip()
st.success(f"Predicted Frame Style: **{predicted_shape.upper()}**")

# ---- Sidebar Filters 
st.sidebar.header("🔎 Filters")
brand_filter = st.sidebar.selectbox("Brand", ["All", "Lenskart", "RayBan", "VincentChase", "Fastrack", "TitanEye+", "Vogue", "JohnJacobs", "Oakley", "Carrera", "Polaroid"])
material_filter = st.sidebar.selectbox("Material", ["All", "Metal", "Acetate", "Plastic"])
price_range = st.sidebar.slider("Price Range (₹)", 1000, 10000, (1500, 6000))

#  Pinecone Filters
filters = {"shape": {"$eq": predicted_shape}}

if brand_filter != "All":
    filters["brand"] = {"$eq": brand_filter}
if material_filter != "All":
    filters["material"] = {"$eq": material_filter}

filters["price"] = {"$gte": price_range[0], "$lte": price_range[1]}

# ---- Text Query as Hard Filters (Multimodal Control) ----
text_lower = text_query.lower()

if "metal" in text_lower:
    filters["material"] = {"$eq": "Metal"}
if "acetate" in text_lower:
    filters["material"] = {"$eq": "Acetate"}
if "plastic" in text_lower:
    filters["material"] = {"$eq": "Plastic"}

if "round" in text_lower:
    filters["shape"] = {"$eq": "Round"}
if "square" in text_lower:
    filters["shape"] = {"$eq": "Square"}
if "aviator" in text_lower:
    filters["shape"] = {"$eq": "Aviator"}
if "wayfarer" in text_lower:
    filters["shape"] = {"$eq": "Wayfarer"}
if "rimless" in text_lower:
    filters["shape"] = {"$eq": "Rimless"}
if "transparent" in text_lower:
    filters["shape"] = {"$eq": "Transparent"}

start_time = time.time()

image_vec = get_image_embedding(image)

if text_query.strip():
    text_vec = get_text_embedding(text_query)
    final_vec = 0.7 * image_vec + 0.3 * text_vec
    final_vec = final_vec / np.linalg.norm(final_vec)
else:
    final_vec = image_vec

# ---- Pinecone Search 
try:
    raw_results = index.query(
        vector=final_vec.tolist(),
        top_k=10,
        include_metadata=True,
        filter=filters
    )
except Exception as e:
    logger.error(f"Pinecone query failed: {e}")
    st.error("Vector search failed.")
    st.stop()

latency = time.time() - start_time
logger.info(f"Query latency: {latency:.2f}s")
st.caption(f"⏱ Search Latency: {latency:.2f} seconds")

# ---- Feedback Boosting 
feedback = load_feedback()
alpha = 0.3
boosted_results = []

for match in raw_results["matches"]:
    base_score = match["score"]
    pid = str(match["id"])   # per-image feedback
    clicks = feedback.get(pid, 0)

    boost = alpha * math.tanh(clicks)  # bounded boost
    match["final_score"] = base_score + boost
    boosted_results.append(match)

boosted_results = sorted(boosted_results, key=lambda x: x["final_score"], reverse=True)


# ---- Display Results 
st.subheader(" Similar Glasses (Strict Shape + Filter Matching)")

if not boosted_results:
    st.warning("No products match the detected style and selected filters.")
else:
    cols = st.columns(5)
    for i, match in enumerate(boosted_results[:10]):
        with cols[i % 5]:
            pid = str(match["id"])
            meta = match["metadata"]

            img_id = pid.zfill(6)
            base_path = f"dataset/images/{img_id}"
            for ext in [".jpg", ".jpeg", ".png"]:
                path = base_path + ext
                if os.path.exists(path):
                    st.image(Image.open(path), width=150)
                    break

            st.write(f"**Brand:** {meta['brand']}")
            st.write(f"**Price:** ₹{meta['price']}")
            st.write(f"**Material:** {meta['material']}")
            st.write(f"**Style:** {meta['shape']}")
            st.write(f"**Similarity:** {match['score']:.3f}")
            st.write(f"**Boosted:** {match['final_score']:.3f}")

            if st.button(" Relevant", key=f"rel_{pid}"):
                fb = load_feedback()
                fb[pid] = fb.get(pid, 0) + 1
                save_feedback(fb)
                st.rerun()

            if st.button(" Not Relevant", key=f"notrel_{pid}"):
                fb = load_feedback()
                fb[pid] = fb.get(pid, 0) - 1
                save_feedback(fb)
                st.rerun()
