# 📄 Intermediate RAG System using Pinecone Vector Database

A production-style **Retrieval-Augmented Generation (RAG)** application that enables users to upload PDF documents, index them into a **Pinecone Vector Database**, and ask natural language questions answered **strictly from the uploaded document content**.

The system is designed to minimize hallucinations by retrieving only the most relevant document chunks before generating responses with a Large Language Model (LLM).

---

## 🌐 Live Demo

🚀 **Streamlit App**

https://ragpineconeq.streamlit.app

---

## 📂 GitHub Repository

https://github.com/wasidkhan7/Intermediate_RAG_PineconeQA

---

# ✨ Features

- 📄 Upload one or multiple PDF documents
- 🧹 Automatic PDF text extraction and cleaning
- ✂️ Intelligent recursive text chunking
- 🧠 Local embedding generation using Sentence Transformers
- 🗂️ Pinecone Vector Database integration
- 🔍 Semantic similarity search
- 🤖 Answer generation using Groq Llama models
- 🚫 Hallucination prevention through context-only prompting
- 📖 Source attribution with:
  - Document name
  - Page number
  - Similarity score
- 🎚 Adjustable chunk size
- 🎚 Adjustable chunk overlap
- 🎚 Adjustable Top-K retrieval
- 🎚 Adjustable similarity threshold
- 📑 Metadata filtering by page number
- 📝 Query history
- 📜 Persistent query logging
- ⚠️ Robust error handling
- 🧩 Clean modular architecture

---

# 🏗 System Architecture

```
                User
                  │
                  ▼
            Upload PDF(s)
                  │
                  ▼
          PDF Text Extraction
                  │
                  ▼
             Text Cleaning
                  │
                  ▼
       Recursive Text Chunking
                  │
                  ▼
      Sentence Transformer
         Embedding Model
                  │
                  ▼
      Pinecone Vector Database
                  │
                  ▼
        Semantic Retrieval
                  │
                  ▼
      Retrieved Context Chunks
                  │
                  ▼
          Groq LLM (Llama)
                  │
                  ▼
     Grounded Answer + Sources
```

---

# 🛠 Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Framework | Streamlit |
| LLM Framework | LangChain |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | Pinecone |
| LLM | Groq (Llama 3.1) |
| PDF Processing | PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| Environment Management | python-dotenv |

---

# 📁 Project Structure

```
Intermediate-RAG-Pinecone/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .env.example
│
├── modules/
│   ├── loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── pinecone_db.py
│   ├── llm_generator.py
│   ├── pipeline.py
│   └── utils.py
│
├── data/
├── tests/
├── architecture/
├── assets/
└── report/
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/wasidkhan7/Intermediate_RAG_PineconeQA.git

cd Intermediate_RAG_PineconeQA
```

Create a virtual environment

```bash
python3.11 -m venv .venv
```

Activate it

Linux/macOS

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root.

```text
PINECONE_API_KEY=

PINECONE_INDEX_NAME=

PINECONE_CLOUD=aws

PINECONE_REGION=us-east-1

GROQ_API_KEY=

GROQ_MODEL=llama-3.1-8b-instant
```

---

# ▶️ Running the Application

```bash
streamlit run app.py
```

---

# 🚀 How It Works

### Step 1

Upload one or more PDF documents.

↓

### Step 2

The application extracts and cleans the document text.

↓

### Step 3

The cleaned text is divided into overlapping chunks.

↓

### Step 4

Each chunk is converted into embeddings using Sentence Transformers.

↓

### Step 5

Embeddings are stored in Pinecone together with metadata.

↓

### Step 6

The user's question is embedded and searched against the vector database.

↓

### Step 7

The most relevant chunks are retrieved.

↓

### Step 8

The retrieved context is sent to the Groq LLM.

↓

### Step 9

The application generates an answer using only the retrieved context and displays the relevant sources.

---

# 🛡 Hallucination Prevention

To improve reliability, the LLM is instructed to answer **only** from the retrieved document context.

If the required information is unavailable, the system responds with:

> **"The answer is not available in the provided document."**

This prevents the model from generating unsupported information.

---

# 📚 Metadata Stored in Pinecone

Each chunk includes:

- Document name
- Page number
- Chunk ID
- Source reference

These metadata fields enable source attribution and metadata-based filtering.

---

# 📊 Implemented Features

- ✅ Multi-document support
- ✅ Adjustable chunk size
- ✅ Adjustable chunk overlap
- ✅ Adjustable Top-K retrieval
- ✅ Adjustable similarity threshold
- ✅ Page filtering
- ✅ Session query history
- ✅ Persistent query logging
- ✅ Source attribution
- ✅ Confidence indication
- ✅ Modular architecture

---

# 🔮 Future Improvements

- Hybrid Search (Dense + Sparse Retrieval)
- OCR support for scanned PDFs
- Conversational memory
- Document deletion from Pinecone
- User authentication
- Docker deployment
- CI/CD pipeline
- Unit and integration tests
- Support for DOCX and TXT files

---

# 📸 Screenshots

Add screenshots here after uploading them to the `assets` folder.

Example

```

<img width="1917" height="946" alt="image" src="https://github.com/user-attachments/assets/24a4acb9-f44f-4510-985d-3386804a003d" />

```

---

# 👨‍💻 Author

**Wasid Khan**

Computer Science Student

University of Peshawar

GitHub:

https://github.com/wasidkhan7

---

# ⭐ If you found this project useful

Please consider giving the repository a ⭐ on GitHub.
