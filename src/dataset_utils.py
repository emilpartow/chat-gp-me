"""
dataset_utils.py

WhatsApp chat data preprocessing: parses message exports, builds context/response pairs, supports block-wise context building,
and logs the entire process.

"""

import glob
import re
import os
import logging
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta
import unicodedata
import re

SYSTEM_MESSAGES = [
    "image omitted", "photo omitted", "sticker omitted", "video omitted",
    "audio omitted", "document omitted", "this message was edited",
    "messages and calls are end-to-end encrypted. only people in this chat can read, listen to, or share them."
]


def setup_logging(log_dir: str = "logs", log_file: str = "prepare_data.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_path}")

def strip_invisible_prefix(line: str) -> str:
    """
    Removes all invisible/control/format unicode chars and whitespace from start of line.
    """
    i = 0
    while i < len(line) and (unicodedata.category(line[i]).startswith('C') or line[i].isspace()):
        i += 1
    return line[i:]

def clean_system_message(text: str) -> str:
    """
    Removes all invisible/control unicode characters and trims whitespace.
    Converts to lower-case. Strips WhatsApp's <...edited...> marker.
    """
    cleaned = ''.join(ch for ch in text if not unicodedata.category(ch).startswith('C'))
    cleaned = cleaned.strip().lower()
    cleaned = re.sub(r"<.*?edited.*?>", "", cleaned).strip()
    return cleaned

def is_system_message(text: str) -> bool:
    """
    Checks if the text is (or starts with) a known English WhatsApp system message.
    """
    t = clean_system_message(text)
    for s in SYSTEM_MESSAGES:
        if t == s or t.startswith(s):
            return True
    return False

def parse_whatsapp_chat(file_path: str, user_identifier: str):
    """
    Parses a WhatsApp chat export file.
    Returns a list of dicts: [{'sender': ..., 'text': ...}, ...]
    """
    logging.info(f"Parsing chat file: {file_path}")
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    pattern = re.compile(r"^\[?(\d{1,2}\.\d{1,2}\.\d{2,4}), (\d{1,2}:\d{2}:\d{2})\] ([^:]+): (.+)$")
    messages = []
    skipped = 0
    for line in lines:
        # ENTSCHEIDEND: entferne invisible/control chars vor dem Regex!
        clean_line = strip_invisible_prefix(line)
        m = pattern.match(clean_line.strip())
        if m:
            date = m.group(1).strip()
            time = m.group(2).strip()
            sender = m.group(3).strip()
            text = m.group(4).strip()
            # Clean edit markers etc.
            text = re.sub(r"<.*?edited.*?>", "", text, flags=re.I).strip()
            if is_system_message(text) or not text:
                skipped += 1
                continue
            messages.append({'date': date, 'time': time, 'sender': sender, 'text': text})
    logging.info(f"Parsed {len(messages)} messages from {file_path}, skipped {skipped} system/empty lines.")
    return messages


def parse_datetime(date_str: str, time_str: str) -> datetime:
    """
    Parses date and time strings into a datetime object.

    Args:
        date_str (str): Date string, e.g. '16.01.25'
        time_str (str): Time string, e.g. '17:44:08'

    Returns:
        datetime: Parsed datetime object.
    """
    for fmt in ("%d.%m.%y %H:%M:%S", "%d.%m.%Y %H:%M:%S"):
        try:
            return datetime.strptime(f"{date_str} {time_str}", fmt)
        except ValueError:
            continue
    raise ValueError(f"Could not parse date/time: {date_str} {time_str}")

def group_message_blocks(messages: List[Dict], max_minutes_gap: int = 3) -> List[Dict]:
    """
    Groups consecutive messages from the same sender within a given time window into message blocks.

    Args:
        messages (List[Dict]): List of parsed messages.
        max_minutes_gap (int): Maximum gap in minutes to still treat as a block.

    Returns:
        List[Dict]: List of message blocks, each block is a dict with sender, text, start_time.
    """
    blocks = []
    prev_sender = None
    prev_time = None
    current_block = {"sender": None, "text": "", "start_time": None}
    for msg in messages:
        sender = msg['sender']
        t = parse_datetime(msg['date'], msg['time'])
        if (
            sender == prev_sender
            and prev_time is not None
            and (t - prev_time).total_seconds() / 60.0 < max_minutes_gap
        ):
            current_block["text"] += " " + msg['text']
        else:
            if current_block["sender"] is not None:
                blocks.append(current_block)
            current_block = {"sender": sender, "text": msg['text'], "start_time": t}
        prev_sender = sender
        prev_time = t
    if current_block["sender"] is not None:
        blocks.append(current_block)
    logging.info(f"Grouped messages into {len(blocks)} blocks.")
    return blocks

def create_context_response_pairs_from_blocks(blocks: List[Dict], user_identifier: str, context_size: int = 2) -> List[Dict]:
    """
    Creates (context, response) pairs using block-wise context.

    Args:
        blocks (List[Dict]): List of message blocks (see group_message_blocks).
        user_identifier (str): Name of the user (USER).
        context_size (int): Number of blocks to use as context.

    Returns:
        List[Dict]: List of dicts with 'context' and 'response'.
    """
    pairs = []
    for i in range(context_size, len(blocks)):
        if blocks[i]['sender'] == user_identifier:
            context_blocks = blocks[i - context_size:i]
            context = ""
            for block in context_blocks:
                context += f"{block['text']}\n"
            response = blocks[i]['text']
            pairs.append({'context': context.strip(), 'response': response})
    logging.info(f"Created {len(pairs)} (context, response) pairs using block-wise context.")
    return pairs

def process_all_chats(
    data_dir: str,
    user_identifier: str,
    context_size: int = 2,
    use_blocks: bool = True,
    max_minutes_gap: int = 3
) -> List[Dict]:
    """
    Processes all WhatsApp .txt files in a directory and generates context/response pairs.

    Args:
        data_dir (str): Directory with WhatsApp .txt exports.
        user_identifier (str): Name of the user.
        context_size (int): Number of messages or blocks as context.
        use_blocks (bool): If True, use block-wise context; else, use message-wise.
        max_minutes_gap (int): Time gap for block segmentation (minutes).

    Returns:
        List[Dict]: All (context, response) pairs from all files.
    """
    all_files = glob.glob(f"{data_dir}/*.txt")
    logging.info(f"Found {len(all_files)} chat files in {data_dir}")
    all_pairs = []
    for file_path in all_files:
        messages = parse_whatsapp_chat(file_path, user_identifier)
        if use_blocks:
            blocks = group_message_blocks(messages, max_minutes_gap=max_minutes_gap)
            pairs = create_context_response_pairs_from_blocks(blocks, user_identifier, context_size)
        else:
            pairs = create_context_response_pairs(messages, user_identifier, context_size)
        all_pairs.extend(pairs)
    logging.info(f"Total pairs from all files: {len(all_pairs)}")
    return all_pairs

def create_context_response_pairs(messages: List[Dict], user_identifier: str, context_size: int = 2) -> List[Dict]:
    """
    (Legacy fallback) Creates (context, response) pairs using last N messages as context.

    Args:
        messages (List[Dict]): List of parsed messages.
        user_identifier (str): Name of the user (USER).
        context_size (int): Number of messages as context.

    Returns:
        List[Dict]: List of dicts with 'context' and 'response'.
    """
    pairs = []
    for i in range(context_size, len(messages)):
        if messages[i]['sender'] == user_identifier:
            context_msgs = messages[i - context_size:i]
            context = ""
            for msg in context_msgs:
                context += f"{msg['text']}\n"
            response = messages[i]['text']
            pairs.append({'context': context.strip(), 'response': response})
    logging.info(f"Created {len(pairs)} (context, response) pairs (message-wise).")
    return pairs

def save_pairs_to_csv(pairs: List[Dict], output_path: str):
    """
    Saves context/response pairs to a CSV file.

    Args:
        pairs (List[Dict]): List of (context, response) dicts.
        output_path (str): Output CSV path.
    """
    df = pd.DataFrame(pairs)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved {len(df)} pairs to {output_path}")