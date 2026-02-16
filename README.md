# ReviewRAG – Context-Aware Restaurant Review QA Agent

## Overview
ReviewRAG is a Retrieval-Augmented Generation (RAG) based question-answering system built to analyze and answer questions from customer reviews of a pizza restaurant.

The system converts restaurant reviews into vector embeddings, stores them in a vector database, retrieves relevant reviews based on user queries, and generates context-aware responses using a lightweight LLM.

---

## Tech Stack

- **LangChain** – Orchestration framework for RAG pipeline
- **Chroma DB** – Vector database for storing embeddings
- **Ollama**
  - `qwen3-embedding:0.6b` – Embedding model
  - `llama3.2:1b` – Generation model
- **Pandas** – Data processing

---

## Project Structure

```
.
├── vector.py      # Creates and persists the vector database
├── main.py        # RAG agent logic (retrieval + prompt + LLM + memory)
├── realistic_restaurant_reviews.csv
└── README.md
```

---

## How It Works

### Vector Creation (`vector.py`)
- Loads restaurant reviews from a CSV dataset
- Converts each review into a LangChain `Document`
- Generates embeddings using `qwen3-embedding:0.6b`
- Stores vectors in Chroma DB
- Creates a retriever (Top-K = 5)

### RAG Agent (`main.py`)
- Retrieves top relevant reviews for a user question
- Injects:
  - Conversation history
  - Retrieved review context
  - User question
- Sends structured prompt to `llama3.2:1b`
- Returns a context-grounded response
- Maintains conversation memory using `ConversationBufferMemory`

---

## Installation

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
```

### 2. Install Dependencies
```bash
pip install pandas langchain_ollama langchain_chroma langchain_core
```

### 3. Ensure Ollama is Running
Make sure Ollama is installed and the required models are pulled:

```bash
ollama pull qwen3-embedding:0.6b
ollama pull llama3.2:1b
```

---

## Running the Project

### Step 1: Create Vector Database
```bash
python vector.py
```

### Step 2: Ask a Question
You can call the RAG function from Python:

```python
from main import run_rag

response = run_rag("What do customers say about delivery time?")
print(response)
```

---

## Example Use Cases

- What are customers saying about pizza quality?
- Are there complaints about delivery delays?
- Which items receive the highest ratings?
- What do customers think about staff behavior?

---

## 🔍 Key Features

✔ Persistent vector storage  
✔ Context-grounded LLM responses  
✔ Conversational memory  
✔ Lightweight local models via Ollama  
✔ Structured RAG architecture  

---

## 📈 Future Improvements

- Add Streamlit or Gradio UI
- Add evaluation pipeline for hallucination detection
