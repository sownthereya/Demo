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
import re

# Load environment variables
load_dotenv()
GOOGLE_API = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API)

# GROQ client setup
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Example benefit codes
benefit_codes = {
    "X1001": "General Consultation",
    "X2002": "Lab Test",
    "X3003": "Radiology",
    "X4004": "Medication Charges",
    "X5005": "Surgical Fees",
}

# Gemini 2.0 Flash model
model = genai.GenerativeModel('gemini-2.0-flash')

# UPDATED Gemini-based PDF image OCR
def get_pdf_text_with_gemini(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf.seek(0)
        images = convert_from_bytes(pdf.read())
        for i, image in enumerate(images):
            response = model.generate_content(
                contents=[
                    image,
                    """
                    Analyze this document page image. Your task is to extract all readable structured billing and customer information.
                    
                    **Priority Extraction Rules**:
                    - Extract text exactly as seen, maintaining line order.
                    - If a line is in **bold, uppercase or larger font at the top**, treat it as a strong **Customer Name** candidate.
                    - Prioritize text that appears:
                        - At the **top-center** or **top-left** of the page.
                        - Just below headers or logos, and not clearly labeled as a seller/company name.

                    Return all text in readable plain format. Label fields where possible:
                    - Customer Name
                    - Document No.
                    - Document Date
                    - Total Amount
                    - Itemized Charges (if present)
                    """
                ]
            )
            extracted = getattr(response, 'text', '[No text extracted]')
            text += extracted + "\n"
    return text

# Table extraction using pdfplumber
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

# Amount extraction
def extract_total_candidates(text):
    amounts = re.findall(r'(?i)(?:total|amount)[^\d]*([\d,]+)', text)
    clean_amounts = [int(a.replace(",", "")) for a in amounts if a]
    return max(clean_amounts) if clean_amounts else None

# Table normalization
def normalize_tables(tables):
    df_list = []
    for table in tables:
        df = pd.DataFrame(table[1:], columns=table[0])
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

# Benefit code mapping
def match_benefits(df):
    if "Code" in df.columns:
        df["Benefit Description"] = df["Code"].map(benefit_codes)
    return df

# Text chunking for embedding
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# Vector store generation
def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("No text found in the PDF.")
        return
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# GROQ-based question answering
def query_groq_llm(context, question):
    prompt = f"""
You are a bilingual (Thai/English) intelligent document extractor, specializing in receipts, invoices, and billing documents.

Extract the following fields with care and layout awareness:

---

**1. Customer Name**

- If there is a **short, bold line (Thai or English)** at the **top center**, or near the header block (sometimes numeric like "5 ชัย"), extract it as the **Customer Name**.
- Do not default to the handwritten block (typically the buyer), unless no better printed candidate exists.
- If there is a **business-like name in bold at top**, use that.
- Avoid names under fields like "Name / ชื่อ", "ที่อยู่", or "Address" — these are often buyers, not customers.

**2. Document Number**
- Look for nearby values labeled as "เลขที่", "Doc No", "Document No", etc.

**3. Document Date**
- Usually near "วันที่", format: DD/MM/YYYY

**4. Total Amount**
- Look for amount near “รวม”, “ยอด”, “Total”, or “Amount” (฿)

---

=== CONTEXT START ===
{context}
=== CONTEXT END ===

QUESTION: {question}

Your task is to identify and return the fields above **based on the real visual layout**. Only return the customer name if you're confident it is bold, printed, and not part of the buyer block.
    """
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024
    )
    return response.choices[0].message.content.strip()
# Process user query
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])
    answer = query_groq_llm(context, user_question)
    st.write("**Data Insights:**", answer)

# Clear input field
def clear_question():
    st.session_state.question_input = ""

# Streamlit app main
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
    # Hide Streamlit file upload size limit message
    st.markdown("""
        <style>
        [data-testid="stFileUploader"] small {
            display: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🔎 DocuGen AI")

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
