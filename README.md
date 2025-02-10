# 📧 Email RAG Assistant

## Overview
The **Email RAG Assistant** is a Streamlit-based application that allows users to fetch, analyze, and query emails using **Retrieval-Augmented Generation (RAG)**. It integrates a **local Ollama model** for AI-powered email analysis and supports vector-based similarity search with **FAISS**.

![image](https://github.com/user-attachments/assets/94fa4b3e-caa6-42e9-bc21-3ee8ff865f3d)


## Features
- 📩 **Fetch Emails**: Connects to Gmail via IMAP and searches emails by keyword.
- 🧠 **AI-Powered Responses**: Uses a local **Ollama** model (e.g., Mistral, Llama3) to analyze email content.
- 🔍 **Vector Search**: Employs **FAISS** for efficient email retrieval based on embeddings.
- 🗑 **Reset Options**: Allows users to clear processed email data while retaining chat history.
- 🎨 **User-Friendly UI**: Built with **Streamlit** for an interactive experience.

## Technologies Used
- **Python** (Streamlit, IMAP, FAISS, BeautifulSoup)
- **LangChain** (for AI model interactions & embeddings)
- **Ollama** (local LLM inference)
- **Google Generative AI Embeddings** (for vectorization)

## Future Enhancements
- 📌 OAuth-based email authentication for better security.
- 🔄 Background email processing with auto-refresh.
- 📊 Email sentiment analysis and summarization.

### Credits

I used the repository of AllAboutAI-YT, link : https://github.com/AllAboutAI-YT/easy-local-rag
