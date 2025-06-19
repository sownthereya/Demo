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
genai.configure(api_key=GOOGLE_API)

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

import re

def extract_total_candidates(text):
    amounts = re.findall(r'(?i)(?:total|amount)[^\d]*([\d,]+)', text)
    clean_amounts = [int(a.replace(",", "")) for a in amounts if a]
    return max(clean_amounts) if clean_amounts else None

# Normalize and map benefit codes
def normalize_tables(tables):
    df_list = []
    for table in tables:
        df = pd.DataFrame(table[1:], columns=table[0])
        df_list.append(df)
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
    You are a bilingual (English + Cantonese) medical document interpretation specialist. The provided content is extracted from multilingual documents, and may include both English and Traditional Chinese (Cantonese), or even be entirely in Cantonese.

    Please follow these rules strictly:

    1. You are a professional medical consultant — **never mention AI, chatbot, or assistant.**  
    2. Always assume the document includes **both Cantonese and English**, and respond accordingly.
    3. For every item or point you list:
    - Present the **English explanation first**
    - Follow immediately with the **Cantonese version in parentheses** (繁體中文).
    - If the content is only in Chinese, return the Cantonese version first and follow with an English translation.
    4. Under the header **“Summary”**, provide your answer in one of these formats:
    - A **clean bullet list** or short paragraphs if the question is general
    - A **clear two-column table** (English / Cantonese) if the question includes phrases like “table format”, “tabular”, etc.
    5. Expand any **benefit codes** from the list below. If a code is unknown, mark it as **“Unknown Code (未知代碼)”**.
    6. Be aware the OCR text may contain noise, formatting issues, or partial values. Extract as clearly as possible, and try to interpret logically.
    7. If **key information is missing**, respond:
    - “I’m sorry, I couldn’t find relevant details in the files provided—I’m limited to the uploaded content. (抱歉，我在您提供的文件中找不到相關資訊——我只能基於您上傳的內容進行回答。)”
    8. If the **question is unclear**, ask a follow-up in both English and Cantonese.
    9. If **Cantonese translation is not possible** for some terms, continue with the English explanation and note it gracefully.

    NOTE:
    - Do not guess or fabricate totals. If multiple totals are found, show them all and identify the highest as “Total (最高總額)”.
    - Never skip language even if it seems redundant — always present **both** English and Cantonese together.

    Benefit Code Mappings:
    - ACF01: Standard Inpatient Food  
    - ACF02: Special Inpatient Food / Medical Nutrition  
    - ACF03: Other Special-Purpose Nutrition  
    - ACF04: Dietary Supplement  
    - ACC13: Mortuary Services  
    - MEAM1: Ambulance Services  
    - ACS01: Other Hospital Charges  
    - ACS02: (Set admit / Gift set)  
    - MED00: Other  

    === CONTEXT ===
    {context}

    === QUESTION ===
    {question}

    === Summary ===
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
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
    # Logo + Title side by side
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        st.image("logo.png", width=100)  # adjust width as needed
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
        # st.caption("Limit 200MB per file")

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