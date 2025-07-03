"""
rag_gradio_chat.py

Starts a local Gradio WhatsApp RAG chatbot.  
Loads your FAISS index, embeddings, and LLM – uses local models if available, otherwise downloads from Huggingface Hub.
Logs all steps and user interactions.

"""

import os
import torch
import logging
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import gradio as gr

# === Setup logging ===
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/chatbot_gradio.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Load FAISS index and WhatsApp texts ===
try:
    index = faiss.read_index("embeddings/whatsapp_faiss.index")
    df = pd.read_csv("embeddings/whatsapp_index_texts.csv")
    texts = [f"{row['context']}\n{row['response']}" for _, row in df.iterrows()]
    logger.info("Loaded FAISS index and WhatsApp texts.")
except Exception as e:
    logger.error(f"Error loading FAISS index or WhatsApp texts: {e}", exc_info=True)
    raise

# === Load embedding model (use local if available, otherwise HF Hub) ===
try:
    local_embed_model = "models/all-MiniLM-L6-v2"
    if os.path.isdir(local_embed_model):
        embed_model_path = local_embed_model
        logger.info(f"Using local embedding model: {embed_model_path}")
    else:
        embed_model_path = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info(f"Local embedding model not found. Using Huggingface: {embed_model_path}")
    embed_model = SentenceTransformer(embed_model_path)
    logger.info("Loaded embedding model.")
except Exception as e:
    logger.error(f"Error loading embedding model: {e}", exc_info=True)
    raise

try:
    # === Load LLM (use local if available, otherwise HF Hub) ===
    local_llm_dir = "models/phi-3-mini-4k-instruct"
    if os.path.isdir(local_llm_dir):
        llm_dir = local_llm_dir
        logger.info(f"Using local LLM: {llm_dir}")
    else:
        llm_dir = "microsoft/phi-3-mini-4k-instruct"
        logger.info(f"Local LLM not found. Using Huggingface: {llm_dir}")

    tokenizer = AutoTokenizer.from_pretrained(llm_dir)
    model = AutoModelForCausalLM.from_pretrained(llm_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    logger.info("Loaded LLM.")
except Exception as e:
    logger.error(f"Error loading LLM: {e}", exc_info=True)
    raise

MAX_INPUT_TOKENS = 256
MAX_OUTPUT_TOKENS = 128

def build_prompt(user_message, retrieved_chunks):
    """
    Combine retrieved WhatsApp-style examples with the user's message
    to create a prompt for the LLM.
    """
    context = "\n---\n".join(retrieved_chunks)
    prompt = f"""You are me, and you answer in the style of my WhatsApp messages. Here are some examples:
{context}
---
Question: {user_message}
Answer:"""
    return prompt.strip()

def retrieve_context(user_message, k=3):
    """
    Use embedding search to find the k most relevant WhatsApp chunks.
    """
    query_vec = embed_model.encode([user_message])
    D, I = index.search(np.array(query_vec), k)
    return [texts[i] for i in I[0]]

def chat(user_message, history):
    """
    Retrieve context and generate a reply using the LLM.
    """
    # Retrieve WhatsApp-style examples
    retrieved_chunks = retrieve_context(user_message)
    prompt = build_prompt(user_message, retrieved_chunks)

    # Tokenize the prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_INPUT_TOKENS
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Generate the answer
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=MAX_INPUT_TOKENS + MAX_OUTPUT_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            top_k=40,
            temperature=0.8,
            no_repeat_ngram_size=3,
            num_beams=1
        )

    # Remove the prompt from the generated text
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    reply = generated[len(prompt):].strip().split("\n")[0].strip()
    # Update chat history (for Gradio display only)
    history = history or []
    history.append((user_message, reply))
    # Log the interaction
    logger.info(f"User: {user_message}\nBot: {reply}\n")
    return history, history

with gr.Blocks(title="chat-gp-me: Your WhatsApp RAG Chatbot") as demo:
    gr.Markdown("## chat-gp-me: Your Personal WhatsApp RAG Chatbot")
    chatbot = gr.Chatbot(label="Chat History")
    state = gr.State([])  # Stores message history for display only
    with gr.Row():
        user_input = gr.Textbox(placeholder="Type your message...", scale=4)
        send_btn = gr.Button("Send", scale=1)
    send_btn.click(chat, [user_input, state], [chatbot, state])
    user_input.submit(chat, [user_input, state], [chatbot, state])

if __name__ == "__main__":
    logger.info("Gradio chatbot session started.")
    demo.launch()
    logger.info("Gradio chatbot session ended.")