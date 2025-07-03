# save_transformer_locally.py
"""
Download and save any SentenceTransformer embedding model locally.
Usage:
    python save_transformer_locally.py --model_name sentence-transformers/all-MiniLM-L6-v2 --save_dir models/all-MiniLM-L6-v2
"""

import os
import argparse
import logging
from sentence_transformers import SentenceTransformer

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/save_transformer.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def save_st_model(model_name, save_dir):
    """
    Downloads and saves a SentenceTransformer embedding model.
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving SentenceTransformer model '{model_name}' to '{save_dir}'")
    try:
        model = SentenceTransformer(model_name)
        model.save(save_dir)
        logger.info("Model saved successfully.")
        print(f"SentenceTransformer model saved to {save_dir}")
    except Exception as e:
        logger.error(f"Error while saving model: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save a SentenceTransformer embedding model locally.")
    parser.add_argument("--model_name", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Model name on Huggingface Hub, e.g. sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--save_dir", type=str,
                        default="models/all-MiniLM-L6-v2",
                        help="Local directory to save the model")
    args = parser.parse_args()
    save_st_model(args.model_name, args.save_dir)