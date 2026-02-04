# 🧠 Mini-RAG: Offline Document Q&A AI

**RAG (Retrieval-Augmented Generation)** application. This tool allows you to chat with your PDF documents using the **Llama 3.1 (8B)** model running entirely on your local machine (GPU).

## 🚀 Features
* **100% Offline:** Runs locally on your hardware (Optimized for RTX 3050 / 6GB VRAM).
* **Zero Cost:** Uses open-source models via Ollama. No APIs required.
* **Privacy First:** Your documents never leave your laptop.
* **Strict Context:** The AI is engineered to answer *only* from your documents, reducing hallucinations (which avoids make-up answers from model's pre-trained knowledge).

## 🛠️ Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/)
* **LLM Engine:** [Ollama](https://ollama.com/) (Llama 3.1 - 8B)
* **Framework:** [LangChain](https://www.langchain.com/)
* **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss) (CPU/Local)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Hardware Acceleration:** NVIDIA RTX 3050 (mobile) GDDR6 6GB VRAM with 95W TGP

---

## 📸 Demo Workflow

### 1. Document Ingestion
Upload your PDFs. The app splits the text into chunks, converts them to vectors, and stores them in a local FAISS index.
<img width="1920" height="1080" alt="Screenshot (71)" src="https://github.com/user-attachments/assets/755f7b14-303b-4d10-a516-ec7353fc12cc" />

### 2. Retrieval & Reasoning
Ask a question. The app performs a similarity search to find the top 4 relevant context chunks and sends them to Llama 3.1.
<img width="1920" height="1080" alt="Screenshot (72)" src="https://github.com/user-attachments/assets/a28015a7-f915-449e-a628-bff87f5d98a5" />

### 3. Generation
The local LLM generates a precise answer based strictly on the retrieved context.
<img width="1920" height="1080" alt="Screenshot (73)" src="https://github.com/user-attachments/assets/fa30a77a-9fab-4db3-bee8-fa2c35508ca5" />

### 4. Hallucination Guard
The system is engineered for strict accuracy. If a user asks a question not covered in the documents, the AI admits it doesn't know rather than making up an answer.
<img width="1920" height="1080" alt="4  Hallucination Guard" src="https://github.com/user-attachments/assets/4d611946-383a-4ba9-8568-f234fbbebf10" />

---

## ⚙️ Installation & Setup

### Prerequisites
1.  **Python 3.8+** installed.
2.  **[Ollama](https://ollama.com/)** installed and running.
3.  **Llama 3.1 Model** pulled locally:
    ```bash
    ollama run llama3.1
    ```

### Step 1: Clone the Repository
```bash
git clone [https://github.com/ATHIF-MD/Mini-RAG-Local.git]
cd Mini-RAG-Local
```

### Step 2: Install Python Dependencies
```bash
pip install streamlit langchain-community langchain-text-splitters faiss-cpu langchain-huggingface
```

### Step 3: Run the App
Make sure Ollama is running in the background, then execute:
```bash
streamlit run app.py
```
