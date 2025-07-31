# 📚 Telegram Study Assistant Bot 🤖

A powerful Telegram bot that transforms uploaded study materials (`PDF`, `DOCX`, `MD`) into intelligent **summaries**, **flashcards** and **answers** using a combination of **FAISS vector search** and **local LLMs via Ollama**.

Ideal for students, educators, and exam-prep warriors who want instant content understanding and revision material generation — all within Telegram.

---

## 🧠 Tech Stack & Libraries Used

| Tool/Library                        | Purpose |
|------------------------------------|---------|
| **Python Telegram Bot (v20.7)**    | Bot interface and command handling |
| **FAISS (Facebook AI)**            | High-speed vector similarity search |
| **sentence-transformers**          | Text embedding using `all-MiniLM-L6-v2` |
| **Ollama + Mistral**               | Local LLM-powered answers and flashcards |
| **PyMuPDF (`fitz`)**               | PDF parsing |
| **python-docx**                    | DOCX text extraction |
| **scikit-learn (`cosine_similarity`)** | Similarity-based ranking |
| **FPDF**                           | Flashcard PDF export |
| **pickle**                         | Vector store persistence |

---

## 🚀 Features

- 📄 Upload `.pdf`, `.docx`, or `.md` files directly in Telegram
- 🧠 Automatically chunks and embeds text for semantic search
- 🔍 Ask questions from the document — smart answers using FAISS + Ollama
- 🧾 Summarize in custom “N points” format
- 🧠 Auto-generate smart flashcards from document content
- 📥 Export flashcards as downloadable PDF
- 🔄 Reset content or download your flashcards anytime
- 🧠 If no document is uploaded, fallback to full LLM QA


---

## 📦 Use This Bot in Your Project

This project is open-sourced under the MIT License.







