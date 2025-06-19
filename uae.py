import streamlit as st
import numpy as np
import pandas as pd
import os
import pdfplumber
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from groq import Groq
from pdf2image import convert_from_bytes
import google.generativeai as genai
import PIL.Image

# Load environment variables
load_dotenv()
GOOGLE_API = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# GROQ client setup
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Benefit code dictionary (example)
benefit_codes = {
    "X1001": "General Consultation",
    "X2002": "Lab Test",
    "X3003": "Radiology",
    "X4004": "Medication Charges",
    "X5005": "Surgical Fees",
}

# Extract plain text using Gemini Vision
model = genai.GenerativeModel('gemini-2.0-flash')

def get_pdf_text_with_gemini(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf.seek(0)
        images = convert_from_bytes(pdf.read())
        for i, image in enumerate(images):
            response = model.generate_content(
                contents=[
                    image,
                    "Extract all readable text and structured billing details from this document page."
                ]
            )
            if hasattr(response, 'text'):
                text += response.text + "\n"
            else:
                text += "[No text extracted]\n"
    return text

# Table extraction from digital PDFs
def extract_tables_from_pdf(pdf_docs):
    tables = []
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table:
                        tables.append(table)
    return tables

# Normalize and map benefit codes
def normalize_tables(tables):
    df_list = []
    for table in tables:
        # original header row
        orig_cols = table[0]
        # make a copy and uniquify
        col_count = {}
        unique_cols = []
        for col in orig_cols:
            count = col_count.get(col, 0) + 1
            col_count[col] = count
            if count > 1:
                unique_cols.append(f"{col}.{count-1}")
            else:
                unique_cols.append(col)
        # build the DataFrame with unique columns
        df = pd.DataFrame(table[1:], columns=unique_cols)
        # reset index so each df has 0,1,2… — avoids index clashes on concat
        df = df.reset_index(drop=True)
        df_list.append(df)
    # if no tables, return empty DataFrame
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

def match_benefits(df):
    if "Code" in df.columns:
        df["Benefit Description"] = df["Code"].map(benefit_codes)
    return df

# Text chunking
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# Create vector store
def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("No text found in the PDF.")
        return
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Groq query
def query_groq_llm(context, question):
    prompt = f"""
You are an intelligent document assistant skilled in extracting structured data from unstructured English text (PDFs, scans, OCR).

🧠 **Your Job**:
Extract the required information strictly based on the provided context. Do not hallucinate. If a field is not explicitly present or clearly inferable, leave its value blank (i.e., after the colon, output nothing).

📄 **Context**:
{context}

🔍 **Question**:
{question}

📋 **Required Fields (in this exact order, each on its own line)**:
1. Bid Category: Insurance or Non-Insurance  
2. Bid Title  
3. Project Description  
4. Project Region  
5. Institution or Ministry (Bidder/Ministry/Institution name)  
6. Submission Deadline  
7. Budget/Amount  
8. Contact Information  
9. Publication Date  
10. Bid Requirements Summary (Bid conditions and requirements)  
11. Line of Business (e.g., Healthcare, Infrastructure, Technology, Energy, Transportation, etc.)  
12. Overall Confidence Score (0-100; estimate confidence in accuracy of the extracted information)

📌 **Rules**:
- Use **only English**.
- **Do not** output Arabic or any other language.
- **Never** output `"Not Found"`. If a field truly cannot be located or confidently inferred, leave it blank after the colon.
- Keep output **clean and minimal**: exactly 12 lines, one per field, in the specified order, in the format:

Now extract the fields based on the context.
"""
    response = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
        top_p=1
    )
    return response.choices[0].message.content.strip()

# Process question
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])
    answer = query_groq_llm(context, user_question)
    st.write("**Data Insights:**", answer)

# Clear input
def clear_question():
    st.session_state.question_input = ""

# Main app

def main():
    st.set_page_config(page_title="DocuGen AI", layout="wide")
    st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #ffffff;
        color: #000000;
    }
    h1, h2, h3 {
        color: #222222;
    }
    .stTextInput>div>div>input {
        background-color: #f9f9f9;
        color: #000000;
        border: 1px solid #ccc;
    }
    .stButton>button {
        background-color: #e0e0e0;
        color: #000000;
        border-radius: 6px;
    }
    .stFileUploader {
        border: 1px dashed #999;
        padding: 1rem;
        border-radius: 10px;
        background-color: #f1f1f1;
    }
    </style>
""", unsafe_allow_html=True)
    st.markdown("""
        <style>
        [data-testid="stFileUploader"] small {
            display: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Logo + Title (side by side)
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        st.image("logo.png", width=100)
    with col_title:
        st.markdown("<h1 style='padding-top: 10px;'>🔎DocuGen AI</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([9, 1], gap="small")
    with col1:
        user_question = st.text_input(
            "",
            value=st.session_state.get("question_input", ""),
            placeholder="Ask a question from the PDF files",
            key="question_input",
            label_visibility="collapsed",
        )
    with col2:
        st.button("×", key="clear_button", help="Clear input", on_click=clear_question)

    if user_question and user_question.strip():
        user_input(user_question)

    with st.sidebar:
        st.markdown("## 📂 Menu")
        st.markdown("Upload your PDF files and click **Submit & Process**")
        pdf_docs = st.file_uploader(
            "Drag and drop files here",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("📥 Submit & Process"):
            with st.spinner("🔍 Processing your PDFs..."):
                raw_text = get_pdf_text_with_gemini(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                tables = extract_tables_from_pdf(pdf_docs)
                if tables:
                    df = normalize_tables(tables)
                    df = match_benefits(df)
                    st.subheader("📋 Extracted Medical Bill Summary")
                    st.dataframe(df)
                st.success("✅ Processing Complete!")

if __name__ == "__main__":
    main()
