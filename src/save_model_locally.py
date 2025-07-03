# save_model_locally.py
"""
Download and save any Huggingface Transformers model (and tokenizer) locally,
with logging. Example usage (with or without arguments):

    python save_model_locally.py
    python save_model_locally.py --model_name microsoft/phi-3-mini-4k-instruct --save_dir models/phi-3-mini-4k-instruct
    python save_model_locally.py --model_name sentence-transformers/all-MiniLM-L6-v2 --save_dir models/all-MiniLM-L6-v2 --embeddings
"""

import os
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# === Setup logging ===
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/save_model.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def save_model_and_tokenizer(model_name, save_dir, is_causal_lm=True):
    """
    Downloads and saves a Huggingface model (and tokenizer) to a local directory.
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving model '{model_name}' to '{save_dir}'")
    try:
        # Download and save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_dir)
        logger.info("Tokenizer saved successfully.")
        # Download and save model
        if is_causal_lm:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(save_dir)
        logger.info("Model saved successfully.")
        print(f"Model and tokenizer saved to {save_dir}")
    except Exception as e:
        logger.error(f"Error while saving model/tokenizer: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save a Huggingface model locally.")
    parser.add_argument("--model_name", type=str,
                        default="microsoft/phi-3-mini-4k-instruct",
                        help="Huggingface model path, e.g. microsoft/phi-3-mini-4k-instruct")
    parser.add_argument("--save_dir", type=str,
                        default="models/phi-3-mini-4k-instruct",
                        help="Local directory to save the model, e.g. models/phi-3-mini-4k-instruct")
    parser.add_argument("--embeddings", action="store_true",
                        help="Set if you are downloading an embedding model, not a text generator (e.g. sentence-transformers)")
    args = parser.parse_args()

    save_model_and_tokenizer(
        args.model_name,
        args.save_dir,
        is_causal_lm=not args.embeddings
    )