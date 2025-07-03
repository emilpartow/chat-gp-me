"""
build_index.py

This script builds a FAISS vector index from your processed WhatsApp chat data, enabling fast semantic search
for Retrieval-Augmented Generation (RAG) applications. It:

1. Loads your (context, response) pairs from 'data/pairs.csv'
2. Converts each entry to a dense embedding using a SentenceTransformer model
   (uses the local model in 'models/all-MiniLM-L6-v2' if available, otherwise downloads from Huggingface Hub)
3. Stores all embeddings in a FAISS index for fast similarity search
4. Saves both the index ('embeddings/whatsapp_faiss.index') and the original texts ('embeddings/whatsapp_index_texts.csv')
5. Logs all steps and errors to 'logs/build_index.log' for easy debugging
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import logging

# === Setup logging ===
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/build_index.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    # === Load your context/response pairs as a DataFrame ===
    pairs_path = os.path.join("data", "pairs.csv")
    df = pd.read_csv(pairs_path)
    logger.info(f"Loaded pairs from {pairs_path} with {len(df)} rows.")

    # === Prepare texts for embedding ===
    texts = [f"{row['context']}\n{row['response']}" for idx, row in df.iterrows()]
    logger.info(f"Prepared {len(texts)} texts for embedding.")

    # === Use local embedding model if available, otherwise fall back to Huggingface Hub ===
    local_model_path = "models/all-MiniLM-L6-v2"
    if os.path.isdir(local_model_path):
        model_path = local_model_path
        logger.info(f"Using local embedding model: {model_path}")
    else:
        model_path = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info(f"Local model not found. Using Huggingface model: {model_path}")

    # === Generate embeddings ===
    model = SentenceTransformer(model_path)
    embeddings = model.encode(texts, show_progress_bar=True)
    logger.info(f"Generated embeddings with shape {embeddings.shape}.")

    # === Build the FAISS vector index ===
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings))
    logger.info(f"Built FAISS index with dimension {d} and {index.ntotal} vectors.")

    # === Save index and texts for later retrieval ===
    os.makedirs("embeddings", exist_ok=True)
    faiss.write_index(index, "embeddings/whatsapp_faiss.index")
    df.to_csv("embeddings/whatsapp_index_texts.csv", index=False)
    logger.info("FAISS index and texts saved to 'embeddings/'.")
    print("FAISS index and texts saved to 'embeddings/'.")
except Exception as e:
    logger.error(f"Error in building index: {e}", exc_info=True)
    print(f"Error: {e}")