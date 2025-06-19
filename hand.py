import google.generativeai as genai
from dotenv import load_dotenv
import os
from pdf2image import convert_from_path
import PIL.Image

# Load API key
load_dotenv()
GOOGLE_API = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API)

# Convert PDF to images
pdf_path = "handy.pdf"  # or any other
images = convert_from_path(pdf_path)  # Each page becomes a PIL.Image

# Initialize Gemini
model = genai.GenerativeModel('gemini-2.0-flash')

# Prompt to extract
prompt = "Extract all the data from this document page."

# Process each page
for i, image in enumerate(images):
    print(f"\n--- Page {i+1} ---")
    response = model.generate_content(
        contents=[image, prompt]
    )
    print(response.text)
