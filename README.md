# 🧠 RAG Tutorial: Retrieval-Augmented Generation with LangChain & FAISS

This project demonstrates a simple **Retrieval-Augmented Generation (RAG)** pipeline using [LangChain](https://www.langchain.com/), OpenAI embeddings, and a local FAISS vector store. It consists of **two parts**:

1. **Storing** – Convert PDF into vector embeddings and store them locally.
2. **Retrieving** – Ask a question and get grounded answers from the stored document using RAG.

---

## 📦 Requirements

Install the dependencies:

```bash
pip install langchain langchain-openai langchain-community langchainhub faiss-cpu pypdf python-dotenv
```

Also create a `.env` file in the project root with:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

---

## 📂 Files in This Project

| File          | Description                                      |
|---------------|--------------------------------------------------|
| `storing.py`  | Loads a PDF, splits it, embeds it, and stores vectors using FAISS |
| `retrieving.py` | Loads FAISS index and answers user questions via RAG |

---

## 📄 Part 1: Embedding & Storing Vectors

### ▶️ `storing.py`

This script does the following:

- Loads a PDF using `PyPDFLoader`
- Splits the text into chunks (1000 characters, 25 overlap)
- Embeds the chunks using OpenAI
- Saves them into a FAISS index locally (`faiss_index_react/`)

### ✍️ Before Running:

Update the path to your PDF:

```python
pdf_path = "/absolute/path/to/example.pdf"
```

Then run:

```bash
python storing.py
```

After running, the FAISS index will be saved locally for later retrieval.

---

## 🔍 Part 2: Query with RAG

### ▶️ `retrieving.py`

This script performs:

- Loading the FAISS vector store
- Using OpenAI to embed the user question
- Searching for relevant chunks (semantic search)
- Using a LangChain prompt template to ask the LLM
- Returning a grounded answer based on your PDF

### Example:

The code includes this query:

```python
response = retrieval_chain.invoke({
    "input": "what is the phone number?"
})
```

Run it:

```bash
python retrieving.py
```

You will get a grounded response from the document stored earlier.

---

## 📌 Notes

- This example uses **FAISS** as a local vector DB for simplicity.
- You can swap it for **Pinecone**, **Chroma**, or other vector stores by changing only a few lines.
- The code uses `allow_dangerous_deserialization=True` when loading FAISS index — only do this in trusted environments.

---

## 🧠 What is RAG?

RAG stands for **Retrieval-Augmented Generation**. The basic flow is:

```
User Question → Embedding → Vector Search → Retrieve Relevant Chunks
→ Augment Prompt with Chunks → Send to LLM → Grounded Answer
```

This technique improves response accuracy and reduces hallucinations in LLM output by grounding answers in real documents.

---

## ✅ Output Example

When you run `retrieving.py`, you will get an answer like:

```
Pinecone is a fully managed cloud-based vector database...
```

Based entirely on the contents of the embedded document.

---

## 🛡️ Security Tip

When using `FAISS.load_local(...)`, be aware of:

```python
allow_dangerous_deserialization=True
```

Only use this flag if you trust the vector index file (avoids Python pickle deserialization attacks).

---

## 🏁 Done!

You're now ready to build more advanced document QA, chatbots, or internal knowledge assistants using LangChain, OpenAI, and vector databases.

Happy building!