import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
import os

# Load model and tokenizer
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint, torch_dtype=torch.float32
)

# Preprocess PDF file and return plain text
def file_preprocessing(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts += text.page_content
    return final_texts[:3000]  # Limit to first 3000 chars to avoid token overflow

# Summarization pipeline using Transformers
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50
    )
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    return result[0]['summary_text']

# PDF viewer in browser using base64
@st.cache_data
def displayPDF(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit App
st.set_page_config(page_title="📄 Document Summarizer", layout="wide")

def main():
    st.title("📄 Document Summarization App using Language Model")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        filepath = f"data/{uploaded_file.name}"
        os.makedirs("data", exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.read())

        if st.button("Summarize"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📘 Uploaded PDF")
                displayPDF(filepath)

            with col2:
                st.subheader("🧠 Generated Summary")
                with st.spinner("Generating summary..."):
                    try:
                        summary = llm_pipeline(filepath)
                        st.success(summary)
                    except Exception as e:
                        st.error(f"Summarization failed: {e}")

if __name__ == "__main__":
    main()
