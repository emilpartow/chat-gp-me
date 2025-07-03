"""
prepare_data.py

Script to create training data pairs from WhatsApp txt exports, using logging and data_config.yaml.

"""

import sys
import os
import yaml
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_utils import (
    setup_logging,
    process_all_chats,
    save_pairs_to_csv
)

def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    log_dir = "logs"
    setup_logging(log_dir)

    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "data_config.yaml"))
    config = load_config(config_path)

    user_identifier = config.get("user_identifier")
    context_size = config.get("context_size", 1)
    max_minutes_gap = config.get("max_minutes_gap", 3) 
    data_dir = config.get("data_dir", "data")
    output_path = config.get("output_path", os.path.join(data_dir, "pairs.csv"))

    if not user_identifier:
        logging.error("No user_identifier found in data_config.yaml. Please add it!")
        sys.exit(1)

    logging.info(f"Using WhatsApp name/identifier: {user_identifier}")
    logging.info(f"Context size (message blocks): {context_size}")
    logging.info(f"Maximum minutes per message block: {max_minutes_gap}")

    pairs = process_all_chats(
        data_dir,
        user_identifier,
        context_size=context_size,
        use_blocks=True,              
        max_minutes_gap=max_minutes_gap
    )
    if not pairs:
        logging.warning("No pairs created. Exiting.")
        sys.exit(1)
    save_pairs_to_csv(pairs, output_path)
    logging.info("Data preparation completed.")

if __name__ == "__main__":
    main()