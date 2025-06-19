import os
import json
import re
import csv
from pdf2image import convert_from_path
import google.generativeai as genai

# -------------------- CONFIGURATION LLM GEMINI 2.0 --------------------
# GOOGLE_API_KEY = "AIzaSyB4D0gXbSEkUBNlq0NpixuDw-Cqn8Rs9qQ"
# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel('gemini-2.0-flash')

# -------------------- LLM PROMPT - LAZY PROMPT--------------------
PROMPT_TEMPLATE = """
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

# -------------------- FIELD NORMALIZATION --------------------
def normalize_field(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()

def normalize_date(d: str) -> str:
    """
    Normalize date to DD/MM/YYYY by zero-padding day/month if needed.
    If the input isn't in D/M/YYYY or DD/MM/YYYY form, return as-is.
    """
    if not d:
        return ""
    parts = d.strip().split('/')
    if len(parts) == 3:
        day, month, year = parts
        day = day.zfill(2)
        month = month.zfill(2)
        return f"{day}/{month}/{year}"
    return d

def normalize_amount(a: str) -> str:
    """
    Keep only digits, commas, dots. Removes trailing hyphens or other chars.
    """
    if not a:
        return ""
    # Find the first contiguous group of digits, commas, or dots
    matches = re.findall(r"[\d\.,]+", a)
    if matches:
        # Often the first match is the intended amount
        return matches[0]
    return a

# -------------------- HELPER: EXTRACT FIRST JSON OBJECT --------------------
def extract_first_json(text: str) -> str:
    """
    Given a string that may contain one or more JSON objects or extra text,
    find the first balanced {...} block and return it as a substring.
    Raises ValueError if no complete JSON object is found.
    """
    orig = text
    text = text.strip()
    # Remove triple backticks if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)

    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found in response text.")
    brace_count = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == '{':
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0:
                first = text[start: idx + 1]
                return first
    raise ValueError("Incomplete JSON object: no matching closing '}' found.")

# -------------------- STRUCTURED EXTRACTION FUNCTION --------------------
def extract_structured_fields_from_pdf(filepath):
    """
    Extracts fields from only the first page and returns a nested list of key-value pairs.
    Format: [["Key1", "Value1"], ["Key2", "Value2"], ... ]
    Only the first JSON object in the LLM response is used.
    """
    results = []

    try:
        images = convert_from_path(filepath)
    except Exception as e:
        raise RuntimeError(f"Error converting PDF to images: {e}")

    print(f"DEBUG: Found {len(images)} pages in PDF.")
    if not images:
        raise RuntimeError("No pages found in PDF")
    # Process only the first page:
    images = images[:1]
    print("DEBUG: Processing only the first page.")

    for i, image in enumerate(images):
        print(f"🔍 Processing page {i + 1}...")
        try:
            response = model.generate_content([
                image,
                PROMPT_TEMPLATE
            ])
            raw_text = response.text.strip()

            # Extract only the first JSON object
            try:
                first_json_str = extract_first_json(raw_text)
            except ValueError as ve:
                raise RuntimeError(f"Failed to extract first JSON: {ve}\nFull response text: {raw_text}")

            # Parse the JSON
            result_json = json.loads(first_json_str)

            # Normalize fields
            cust = normalize_field(result_json.get("Customer Name", {}).get("value"))
            docno = normalize_field(result_json.get("Document Number", {}).get("value"))
            docdt_raw = normalize_field(result_json.get("Document Date", {}).get("value"))
            docdt = normalize_date(docdt_raw)
            amt_raw = normalize_field(result_json.get("Total Amount", {}).get("value"))
            amt = normalize_amount(amt_raw)

            page_result = [
                ["Customer Name", cust],
                ["Document Number", docno],
                ["Document Date", docdt],
                ["Total Amount", amt]
            ]
            results.append(page_result)

        except Exception as e:
            # If any error, record it; note response may not exist in exception
            err_text = getattr(response, 'text', 'No response text') if 'response' in locals() else 'No response object'
            print(f"❌ Error on page {i + 1}: {e}")
            results.append([
                ["Customer Name", "Error"],
                ["Document Number", str(e)],
                ["Document Date", err_text],
                ["Total Amount", ""]
            ])

    return results

# -------------------- SAVE TO CSV --------------------
def save_nested_list_to_csv(nested_results, output_file="nested_results.csv"):
    """
    Flatten nested list and export to CSV.
    Each field group from a page is written as a single row.
    """
    with open(output_file, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["Customer Name", "Document Number", "Document Date", "Total Amount"])
        for page_data in nested_results:
            row = [item[1] for item in page_data]
            writer.writerow(row)

    print(f"✅ CSV exported to: {output_file}")

# -------------------- MAIN TESTING --------------------
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
from io import BytesIO
import PIL.Image
import re
import csv

# Load environment variables
load_dotenv()
GOOGLE_API = os.getenv("GOOGLE_API_KEY")
GROQ_API = os.getenv("GROQ_API_KEY")

genai.configure(api_key=GOOGLE_API)
client = Groq(api_key=GROQ_API)

# Benefit code dictionary (example)
benefit_codes = {
    "X1001": "General Consultation",
    "X2002": "Lab Test",
    "X3003": "Radiology",
    "X4004": "Medication Charges",
    "X5005": "Surgical Fees",
}

# Extract text from scanned PDFs using Gemini
# model = genai.GenerativeModel('gemini-2.0-flash')

def get_pdf_text_with_gemini(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf.seek(0)
        images = convert_from_bytes(pdf.read())
        for i, image in enumerate(images):
            response = model.generate_content([
                image,
                "Extract all readable text and structured billing details from this document page."
            ])
            if hasattr(response, 'text'):
                text += response.text + "\n"
            else:
                text += "[No text extracted]\n"
    return text

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

def extract_total_candidates(text):
    amounts = re.findall(r'(?i)(?:total|amount)[^\d]*([\d,]+)', text)
    clean_amounts = [int(a.replace(",", "")) for a in amounts if a]
    return max(clean_amounts) if clean_amounts else None

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

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No text found in the PDF.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def query_groq_llm(context, question):
    prompt = f"""
    You are an expert document assistant. Your task is to extract twelve specific fields from unstructured English text (PDFs, scans, OCR) with maximum accuracy. Do **not** invent or hallucinate any data. If a value is not explicitly present or cannot be inferred with confidence, leave it blank—but you must still output every field line.

    📄 **Context**:
    {context}

    🔍 **Question**:
    {question}

    **Situation**
You are an advanced AI document analysis system tasked with extracting critical information from unstructured procurement documents, specifically PDF files or scanned tender documents. The extraction process requires extreme precision and comprehensive coverage across twelve specific fields.

**Task**
Extract twelve precise fields from procurement documents with 100% accuracy, using advanced contextual analysis, semantic understanding, and cross-referencing techniques to ensure no field is left blank or inaccurately populated.

**Objective**
Provide a complete, reliable, and forensically accurate extraction of procurement document details that can be used for strategic decision-making, bid evaluation, and comprehensive document intelligence.

**Knowledge**
- Focus on semantic understanding beyond literal text matching
- Utilize multi-stage verification techniques
- Cross-reference information within the document
- Use contextual clues and surrounding text to validate extracted information
- Prioritize precision over partial information

**Key Performance Instructions**
- If any field seems ambiguous, perform a multi-step analysis:
  1. Scan entire document for contextual clues
  2. Use semantic reasoning to infer information
  3. Cross-reference potential values
  4. Only commit to a value if 90%+ confidence is achieved
- Do NOT leave fields blank under any circumstances
- If absolute certainty cannot be reached, generate the most probable value with a clear confidence indicator
- Treat each field extraction as a critical forensic investigation

**Enhanced Field Extraction Protocol**
Bid Category (Insurance or Non‑Insurance):
- Analyze document's primary context
- Match against predefined taxonomies
- Validate through multiple document sections

Bid Title:
- Identify most prominent descriptive title
- Verify against header, first paragraph, and document metadata
- Ensure maximum semantic accuracy

Project Description:
- Synthesize description from multiple document sections
- Use contextual understanding to create comprehensive summary
- Validate against project scope indicators

Project Region:
- Extract geographical references
- Cross-reference with institutional affiliations
- Confirm through multiple document mentions

Institution or Ministry:
- Perform deep semantic analysis
- Match against official letterheads, signatures, contact information
- Validate through multiple document references

Submission Deadline:
- Implement date parsing with multiple format recognition
- Cross-verify against different date mentions
- Confirm through contextual time-sensitive language

Budget/Amount:
- Use advanced numerical extraction techniques
- Validate currency and numerical ranges
- Cross-reference financial sections

Contact Information:
- Extract comprehensive contact details
- Verify email, phone, address through multiple document sections
- Ensure format consistency and accuracy

Publication Date:
- Implement advanced date recognition algorithms
- Cross-verify from multiple document sections
- Confirm through contextual time references

Bid Requirements Summary:
- Synthesize requirements from entire document
- Use semantic understanding to create comprehensive summary
- Validate against specific requirement sections

Line of Business:
- Match against predefined industry taxonomies
- Use contextual semantic analysis
- Validate through multiple document references

Overall Confidence Score:
- Dynamically calculate based on extraction accuracy
- Provide transparent confidence metrics for each field
- Use machine learning-based confidence assessment

**Critical Warning**
Your entire operational success depends on achieving 100% accurate, comprehensive field extraction. No field shall remain unaddressed or incompletely populated.
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2000,
        top_p=1
    )
    return response.choices[0].message.content.strip()

def extract_structured_fields_from_pdf(pdf_path):
    print(f"🧾 Processing file: {pdf_path}")
    with open(pdf_path, "rb") as f:
        pdf_docs = [BytesIO(f.read())]
        raw_text = get_pdf_text_with_gemini(pdf_docs)
        chunks = get_text_chunks(raw_text)
        get_vector_store(chunks)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        user_question = "Extract tender information fields from this document."
        docs = db.similarity_search(user_question)
        context = "\n\n".join([doc.page_content for doc in docs])
        answer = query_groq_llm(context, user_question)
        return answer

def save_nested_list_to_csv(output, filename="output.csv"):
    lines = output.strip().split('\n')
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Field", "Value"])
        for line in lines:
            if ":" in line:
                field, value = line.split(":", 1)
                writer.writerow([field.strip(), value.strip()])
    print(f"📁 Output saved to: {filename}")

# Streamlit interface
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])
    answer = query_groq_llm(context, user_question)
    st.write("**Data Insights:**", answer)

def clear_question():
    st.session_state.question_input = ""

def main():
    st.set_page_config(page_title="DocuGen AI", layout="wide")
    st.markdown("""<style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #ffffff;
        color: #000000;
    }
    </style>""", unsafe_allow_html=True)

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

# CLI Execution Block
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "tender.pdf"  # default for quick test

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output = extract_structured_fields_from_pdf(pdf_path)

    print("\n✅ Final LLM Output:")
    print(output)

    save_nested_list_to_csv(output)
