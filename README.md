# YouTube RAG Chatbot

An intelligent question-answering system for YouTube videos using Retrieval-Augmented Generation (RAG).

## Features
- 🎥 Automatic transcript extraction from YouTube videos
- 🔍 Semantic search using FAISS vector store
- 💾 Persistent vector storage (no re-processing needed)
- 🤖 Context-aware responses powered by Mistral-7B
- ⚡ Efficient chunk-based retrieval with HuggingFace embeddings

## Tech Stack
- **LangChain** - RAG pipeline orchestration
- **FAISS** - Vector similarity search
- **HuggingFace** - Embeddings (BGE) & LLM (Mistral-7B)
- **youtube-transcript-api** - Transcript fetching

## Use Cases
- Educational content summarization
- Research on video content
- Accessibility for video information
- Quick fact-checking from long videos
