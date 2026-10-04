# Document Q&A Chatbot (RAG)

Ask questions over an uploaded document using a simple **Retrieval-Augmented Generation (RAG)** pipeline: chunk → retrieve → answer.

## Live Demo

Deploy free on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Open this repo in Streamlit Cloud  
2. Main file: `app.py`  
3. Deploy → public URL (e.g. `https://document-qa-chatbot-xxxx.streamlit.app`)

## Features

- Upload `.txt` or paste document text
- Chunking with overlap
- Keyword-based retrieval (top-k passages)
- Extractive answers in demo mode (no API key required)
- Easy to extend with OpenAI / Groq for generative answers

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Try with the included `sample_doc.txt`.

## Tech Stack

- Python
- Streamlit
- RAG pattern (chunk + retrieve + answer)

## Author

**Gowtham V** · [github.com/gowtham-v08](https://github.com/gowtham-v08) · [Portfolio](https://gowtham-v.vercel.app)
