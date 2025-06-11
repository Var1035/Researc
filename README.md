<h1 align="center">📄 Document Summarizer App</h1>
<h3 align="center">Built with Streamlit, LangChain, Hugging Face Transformers, and PyTorch</h3>

---

📌 **Project Overview**

This is an AI-powered **PDF Document Summarizer App** that extracts text from PDF files and generates concise summaries using the `LaMini-Flan-T5` model. The app is built with **Streamlit** for an interactive web interface and integrates **LangChain** for smart text chunking and **Transformers** for natural language processing.

---

🚀 **Key Features**

- 📄 Upload and preview PDF documents directly in-browser  
- ✂️ Intelligent document chunking using LangChain  
- 🤖 Summarization using `LaMini-Flan-T5` from Hugging Face  
- ⚡ Efficient text preprocessing to avoid token overflow  
- 🧠 Built-in PDF viewer for side-by-side comparison  
- 🌐 Streamlit-powered UI for fast deployment

---

🧑‍💻 **Tech Stack**

- **Frontend**: Streamlit  
- **NLP**: Hugging Face Transformers (`pipeline` API), LaMini-Flan-T5  
- **Text Preprocessing**: LangChain (`RecursiveCharacterTextSplitter`)  
- **PDF Parsing**: PyPDFLoader (LangChain Community)  
- **Frameworks**: PyTorch  
- **Other Tools**: Base64 encoding for PDF rendering

---

📂 **How It Works**

1. 📤 Upload a `.pdf` file via the Streamlit interface  
2. 📚 Text is extracted and chunked using `RecursiveCharacterTextSplitter`  
3. 🤖 The summarization pipeline runs with `T5ForConditionalGeneration`  
4. 📄 The original PDF and the generated summary are displayed side by side

---

🛠️ **Installation & Setup**

```bash
# Clone the repository
git clone https://github.com/yourusername/document-summarizer-app.git
cd document-summarizer-app

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install required packages
pip install -r requirements.txt
