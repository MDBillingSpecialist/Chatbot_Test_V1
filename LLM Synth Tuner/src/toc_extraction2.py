import fitz  # PyMuPDF for faster PDF processing
import logging
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# Setup logging configuration
logging.basicConfig(
    filename='basic_toc_extraction.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def load_openai_client():
    """Load OpenAI client with API key from environment."""
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OpenAI API key not found. Please set it in the .env file.")
        return None
    return OpenAI(api_key=api_key)

def extract_first_10_pages(pdf_path):
    """Extract the first 10 pages of the PDF using PyMuPDF for faster processing."""
    text_chunk = ""
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(min(10, pdf_document.page_count)):  # Extract up to 10 pages
            page = pdf_document.load_page(page_num)
            text_chunk += page.get_text("text") + "\n"
        pdf_document.close()
        return text_chunk
    except Exception as e:
        logging.error(f"Error extracting text from PDF with PyMuPDF: {e}")
        return ""

def extract_toc_llm(client, text_chunk):
    """Request TOC extraction in JSON format from the LLM."""
    logging.info("Sending request to OpenAI to analyze TOC...")

    prompt = (
        "You are analyzing a corporate handbook or business document. This document contains clearly defined sections, subsections, "
        "and page numbers in its Table of Contents (TOC). Your task is to extract all topics, subtopics, and corresponding "
        "page numbers. If page numbers are not provided, make a note. Please return the result in a structured JSON format, "
        "organized by section and subsection, with clear page numbers where possible."
        "\n\nDocument content:\n\n" + text_chunk
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4096
        )

        raw_toc_string = response.choices[0].message.content
        logging.debug(f"Raw TOC string content: {raw_toc_string}")

        toc_json_start = raw_toc_string.find("{")
        toc_json_end = raw_toc_string.rfind("}") + 1
        toc_json_string = raw_toc_string[toc_json_start:toc_json_end]

        toc_json = json.loads(toc_json_string)
        logging.info("LLM TOC extraction completed successfully.")
        return toc_json

    except Exception as e:
        logging.error(f"Error during LLM TOC extraction: {e}")
        return None

def save_toc_to_json(toc_data, output_file):
    """Save the extracted TOC into a JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(toc_data, f, indent=4)
        logging.info(f"TOC saved to JSON successfully at {output_file}.")
        print(f"TOC saved to: {output_file}")
    except Exception as e:
        logging.error(f"Error saving JSON file: {e}")

def main():
    pdf_path = r'C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\data\raw\Handbook.PDF'
    toc_output_file = os.path.join("data", "toc_output.json")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(toc_output_file), exist_ok=True)
    
    # Load OpenAI client
    client = load_openai_client()
    if not client:
        logging.error("OpenAI client could not be loaded. Exiting.")
        return
    
    # Step 1: Extract the first 10 pages of the PDF
    text_chunk = extract_first_10_pages(pdf_path)
    if not text_chunk:
        logging.error("No text extracted from the first 10 pages.")
        return
    
    # Step 2: Send the text to the LLM for TOC extraction
    toc_data = extract_toc_llm(client, text_chunk)
    if toc_data:
        # Step 3: Save the extracted TOC to a JSON file
        save_toc_to_json(toc_data, toc_output_file)
    else:
        logging.error("TOC extraction failed or incomplete.")

    logging.info("TOC extraction process completed.")

if __name__ == "__main__":
    main()
