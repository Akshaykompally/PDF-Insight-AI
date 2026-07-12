# 📄 PdfGene

An AI-powered PDF Question Answering application built using **Mistral AI**, **LangChain**, **ChromaDB**, and **Streamlit**. Upload any PDF and ask questions in natural language. The application retrieves the most relevant content from the document using Retrieval-Augmented Generation (RAG) and generates accurate answers based only on the uploaded PDF.

---

## 🚀 Features

* 📂 Upload PDF documents
* 🤖 Ask questions in natural language
* 🔍 Semantic search using ChromaDB
* 🧠 Retrieval-Augmented Generation (RAG)
* ⚡ Fast document retrieval with MMR search
* 💬 Interactive chat interface using Streamlit
* 🔒 API keys managed securely with a `.env` file

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Backend

* Python
* LangChain

### AI Model

* Mistral AI (`mistral-small-2506`)

### Embeddings

* Mistral AI Embeddings

### Vector Database

* ChromaDB

### Document Loader

* PyPDFLoader

### Text Splitting

* RecursiveCharacterTextSplitter

---

## 📂 Project Structure

```text
PDF-Insight-AI/
│── main.py
│── requirements.txt
│── .env                # Not included in GitHub
│── .gitignore
│── README.md
│── chroma_db/          # Generated automatically
│── temp/               # Temporary uploaded PDFs
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Akshaykompally/PdfGene.git
```

```bash
cd PdfGene
```

### 2. Create a virtual environment

Windows

```bash
python -m venv .venv
```

Activate

```bash
.venv\Scripts\activate
```

macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
MISTRAL_API_KEY = your_mistral_api_key
```

> **Note:** Never commit your `.env` file to GitHub.

---

## ▶️ Run the Application

```bash
streamlit run main.py
```

or

```bash
python -m streamlit run main.py
```

---

## 📸 How It Works

1. Upload a PDF document.
2. The PDF is split into text chunks.
3. Chunks are converted into embeddings using Mistral AI.
4. Embeddings are stored in ChromaDB.
5. Relevant chunks are retrieved using semantic search.
6. Mistral AI generates answers using only the retrieved PDF context.

---

## 📋 Requirements

* Python 3.10+
* Mistral AI API Key

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 📦 Dependencies

* Streamlit
* LangChain
* LangChain Community
* LangChain Core
* LangChain Text Splitters
* LangChain MistralAI
* ChromaDB
* PyPDF
* Python Dotenv

---

## 🔮 Future Improvements

* Support multiple PDF uploads
* Conversation memory
* PDF summarization
* Citation of source pages
* Chat history export
* Dark mode UI
* OCR support for scanned PDFs

---

## 👨‍💻 Author

**Akshay Kompally**

GitHub: https://github.com/Akshaykompally

---

## 📄 License

This project is licensed under the MIT License.
