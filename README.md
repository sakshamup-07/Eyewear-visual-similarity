# 👓 Lenskart Visual Similarity Search (AI Powered)

## Project Overview
This project implements a multimodal visual similarity search system for eyewear, inspired by production-grade pipelines used in e-commerce platforms such as Lenskart. The system allows users to upload an image of glasses and retrieve visually similar products using deep learning and vector search, with optional text refinement and user feedback learning.

---

## Architecture

### Layers

1. **UI Layer**
   - Streamlit web application
   - Image upload, text input, filters, and feedback buttons

2. **AI Inference Layer**
   - CLIP (openai/clip-vit-base-patch32)
   - Image embedding
   - Text embedding
   - Zero-shot attribute prediction (shape, frame type)

3. **Vector Storage Layer**
   - Pinecone vector database
   - Stores embeddings + metadata (brand, price, material, shape)

4. **Re-Ranking Layer**
   - Feedback-based score boosting
   - Learning user preferences over time

5. **Observability Layer**
   - Python logging
   - Latency tracking
   - Failure monitoring

---

## Model Choice

### CLIP (Contrastive Language–Image Pretraining)

We use `openai/clip-vit-base-patch32` because:

- Joint image-text embedding space
- Enables multimodal search
- Supports zero-shot classification
- Produces 512-dimensional semantic vectors
- Industry-standard for vision retrieval

Used for:
- Image feature extraction
- Text query embedding
- Frame attribute detection (square, round, aviator, etc.)

---

## Vector Similarity Metric

### Cosine Similarity (Pinecone Index)

Cosine distance is chosen because:

- CLIP embeddings are L2-normalized
- Measures semantic direction, not magnitude
- Performs better than Euclidean in high-dimensional spaces
- Standard metric for vision-language models

Index configuration:
- Metric: Cosine
- Dimensions: 512
- Type: Dense
- Deployment: On-demand

---

## Multimodal Query Fusion

Final query vector is computed as:

```
final_vector = 0.7 * image_embedding + 0.3 * text_embedding
```

This allows:
- Visual dominance
- Text-based refinement
- Robust semantic matching

---

## Attribute Detection (Zero-Shot)

CLIP zero-shot classification is used to detect:

- Square frame
- Round frame
- Aviator
- Wayfarer
- Rimless
- Transparent frame

Top-scoring label is used as a **strict Pinecone metadata filter** to ensure only matching shapes are retrieved.

---

## Feedback Learning

User interactions:

- 👍 Relevant
- 👎 Not Relevant

Stored in:
```
style_feedback.json
```

Re-ranking formula:
```
final_score = similarity_score + alpha * feedback_count
```

This enables:
- Personalization
- Online preference learning
- Ranking adaptation

---

## Observability & Logging

Logging is implemented using Python’s `logging` module.

Log file:
```
app.log
```

Captured events:
- Corrupt image uploads
- CLIP inference failures
- Pinecone query failures
- Search latency
- High-latency warnings (>2 seconds)

This satisfies:
- Failure observability
- Performance monitoring
- Debug traceability

---

## Color Detection (Experimental Module)

A classical computer-vision color detection module was attempted using:

- Dominant color clustering (RGB / HSV)
- Frame region averaging
- Color name mapping

Limitations encountered:

- Background dominance
- Lighting sensitivity
- No segmentation of frame vs lens vs background
- No labeled color ground truth in dataset

Due to unreliable accuracy, the color module was **documented but excluded from strict filtering**, and CLIP semantic retrieval was retained as the primary signal.

This mirrors real-world systems where:
- Color is derived from catalog metadata
- Vision-only color inference is secondary

---

## How to Run

### 1. Install Dependencies

```bash
pip install streamlit torch transformers pinecone-client pillow numpy
```

### 2. Create Secrets File

Path:
```
.streamlit/secrets.toml
```

Content:
```toml
PINECONE_API_KEY = "your_api_key"
PINECONE_INDEX = "lenskart-eyewear"
```

### 3. Run Application

```bash
streamlit run app.py
```

---

## Performance Characteristics

- CLIP inference: GPU-accelerated when available
- Vector search: Sub-second (Pinecone cosine ANN)
- End-to-end latency: ~1–3 seconds typical
- Observed slow cases logged for diagnosis

---

## Industry Alignment

The pipeline architecture mirrors production systems used by:

- Lenskart
- Amazon Visual Search
- Google Shopping Lens
- Flipkart Image Search

Key industry practices demonstrated:

- Multimodal embedding
- Vector databases
- Metadata filtering
- Feedback loops
- Observability
- Separation of inference and storage layers
- Strict ranking control

---

## Summary

This project demonstrates an end-to-end AI visual search system with:

- CLIP-based multimodal understanding
- Pinecone vector retrieval
- Zero-shot attribute reasoning
- User-driven ranking adaptation
- Latency monitoring and logging
- Modular, scalable architecture suitable for real-world deployment
