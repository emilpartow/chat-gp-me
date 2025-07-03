# chat-gp-me 🤖

**chat-gp-me** is your personal AI chatbot that learns to reply in your style — using your private WhatsApp chats as a knowledge base!
It leverages modern Retrieval-Augmented Generation (RAG) and state-of-the-art transformer models, all running locally on your machine.

---

## 🚀 What is it?

- Transforms your WhatsApp chat history into a unique AI chatbot that answers *like you*.
- Uses your chat data as a live "memory" with vector search (RAG).
- Lets you use local Huggingface models (LLMs and embeddings), with optional downloading helper.
- Includes robust data preprocessing, embedding, and a modern Gradio web chat demo.

Everything happens locally: your chats never leave your computer.

---

## 📁 Project Structure

```
chat-gp-me/
│
├── data/
│ ├── raw/
│ │ ├── chat_1.txt # Exported WhatsApp chat .txt files (originals)
│ │ └── ...
│ │
│ ├── data_config.yaml # Configuration of data processing (via src/prepare_data.py)
│ └── pairs.csv # Processed (context,response)-pairs for RAG
│
├── embeddings/ # Vector index and related CSVs
│ ├── whatsapp_faiss.index
│ └── whatsapp_index_texts.csv
│
├── models/ # Local Huggingface model folders (LLMs and embeddings)
│ ├── phi-3-mini-4k-instruct/
│ └── all-MiniLM-L6-v2/
│
├── logs/ # All logs (preprocessing, chatbot, download helper, ...)
│
├── src/
│ ├── prepare_data.py # Preprocessing WhatsApp txt to (context, response) CSV
│ ├── build_index.py # Build FAISS index for embedding search
│ ├── rag_chat.py # Gradio demo app (RAG chatbot)
│ ├── save_model_locally.py # Download any Huggingface LLM locally
│ ├── save_transformer_locally.py # Download any SentenceTransformer model locally
│ └── dataset_utils.py # WhatsApp parsing and helpers
│
├── requirements.txt
├── README.md
└── LICENSE
```
---

## 🛠 How it works

### 1. Export your WhatsApp chat
To use this tool, you need to export one or more WhatsApp chats as plain text:

1. **Open WhatsApp** and select the chat to export.
2. Tap the menu (`⋮`) > **More** > **Export chat** (Android) or tap the chat name > **Export Chat** (iOS).
3. Choose **Without media**.
4. Save the exported `.txt` file to your computer.
5. Place all file(s) into the `data/raw/` folder of this project.

*You can export as many chats as you like. All `.txt` files in the `data/raw/` folder will be processed.*

**Please note:**  
At this stage, the project **only supports WhatsApp chat exports in English language/region**.

Your exported chat files should look like this:
```
[16.01.25, 18:37:31] John: Message text
[16.01.25, 18:37:35] Jane: Another message
```

#### How to change your WhatsApp name

WhatsApp exports every message with the sender’s name as it appears in your chats.  
If your name is not unique among your chat partners (e.g., there are two people named "Alex"),  
you **must** give yourself a unique name or identifier in WhatsApp **before** exporting your chats.

1. Open WhatsApp > Settings (Profile).
2. Tap your profile name, and enter a unique identifier.
3. Export your chats with this new name as described above.

> ⚠️ *If more than one person in your chats shares your name, this step is required for accurate processing!*

---

### 2. Preprocessing

Use `src/prepare_data.py` to parse your WhatsApp txt files and create context/response pairs (`data/pairs.csv`).

```bash
python src/prepare_data.py
```

---

### 3. Download models locally (recommended for speed & offline use)

Use the helper scripts to download models for LLM and embedding:
```bash
python src/save_model_locally.py
python src/save_transformer_locally.py
```

---

### 4. Build the embedding index

Convert all your (context, response) pairs into embeddings and build the FAISS index:

```bash
python src/build_index.py
```

---

### 5. Start the chatbot demo

Run the Gradio RAG chatbot app locally:

```bash
python src/rag_chat.py
```
Open the Gradio link shown in your terminal to chat with your own "chat-gp-me"!

---

## 🔒 Privacy & Security

All processing happens locally on your own machine.
Your chat data is never uploaded or shared.

---

## 📋 License

[MIT License](LICENSE)

---

## 📬 Contact & feedback

Feel free to open issues or pull requests if you have ideas, questions, or find bugs!

