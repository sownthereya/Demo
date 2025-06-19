import os
import json
import re
import csv
from pdf2image import convert_from_path
import google.generativeai as genai

# -------------------- CONFIGURATION: Gemini 2.0 --------------------
GOOGLE_API_KEY = "AIzaSyB4D0gXbSEkUBNlq0NpixuDw-Cqn8Rs9qQ"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# -------------------- PROMPT TEMPLATE --------------------
PROMPT_TEMPLATE = """
You are a highly accurate Thai document parser. Extract the following structured fields from the PDF page image:

1. "Customer Name" — Focus on Thai text; extract full, accurate name
2. "Document Number" — Exact number format as in the document
3. "Document Date" — Use format 'DD/MM/YYYY' if available
4. "Total Amount" — Extract the final amount in Baht (฿), including commas if present

Return your result ONLY in this exact JSON format (no extra fields or comments):

{
  "Customer Name": { "value": "..." },
  "Document Number": { "value": "..." },
  "Document Date": { "value": "..." },
  "Total Amount": { "value": "..." }
}

⚠️ Do NOT include extra headers, explanations, or Markdown code blocks (e.g., no ```json).
Ensure all field values are accurate and clearly readable.
"""

# -------------------- FIELD NORMALIZATION --------------------
def normalize_field(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()

# -------------------- PDF EXTRACTION FUNCTION --------------------
def extract_structured_fields_from_pdf(filepath):
    """
    Extracts key fields from a PDF using Gemini 2.0.
    Returns a flat list of [ ["Key", "Value"], ... ]
    """
    results = []

    try:
        images = convert_from_path(filepath)
    except Exception as e:
        raise RuntimeError(f"Error converting PDF to images: {e}")

    for i, image in enumerate(images):
        print(f"🔍 Processing page {i + 1}...")

        try:
            response = model.generate_content([
                image,
                PROMPT_TEMPLATE
            ])

            # Clean JSON from response
            clean_text = re.sub(r"^```(json)?\s*|\s*```$", "", response.text.strip(), flags=re.IGNORECASE)
            result_json = json.loads(clean_text)

            # Normalize and append results
            results.extend([
                ["Customer Name", normalize_field(result_json.get("Customer Name", {}).get("value"))],
                ["Document Number", normalize_field(result_json.get("Document Number", {}).get("value"))],
                ["Document Date", normalize_field(result_json.get("Document Date", {}).get("value"))],
                ["Total Amount", normalize_field(result_json.get("Total Amount", {}).get("value"))]
            ])

        except Exception as e:
            print(f"❌ Error on page {i + 1}: {e}")
            results.extend([
                ["Customer Name", "Error"],
                ["Document Number", str(e)],
                ["Document Date", getattr(response, 'text', 'No response text')],
                ["Total Amount", None]
            ])

    return results

# -------------------- SAVE TO CSV --------------------
def save_results_to_csv(results, output_path="extracted_data.csv"):
    headers = ["Field", "Value"]

    with open(output_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)

    print(f"\n📁 Results saved to: {output_path}")

# -------------------- MAIN --------------------
if __name__ == "__main__":
    pdf_path = "thai2.pdf"

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"❗ PDF not found: {pdf_path}")

    output = extract_structured_fields_from_pdf(pdf_path)

    print("\n✅ Final Extracted Output (Key-Value Pairs):")
    for key, value in output:
        print(f"{key}: {value}")

    save_results_to_csv(output)
