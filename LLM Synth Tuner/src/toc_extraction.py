import re
import fitz  # PyMuPDF for faster PDF processing
import logging
import json
import os
import asyncio
import time
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Setup logging configuration
logging.basicConfig(
    filename='plan_maker.log',
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

def dynamic_chunk_size(total_pages):
    """Determine the chunk size dynamically based on the total number of pages."""
    if total_pages <= 50:
        return 2  # Small document
    elif total_pages <= 300:
        return 5  # Medium document
    else:
        return 10  # Large document

def detect_headers(text):
    """Detect headers based on regular expressions and text structure."""
    headers = re.findall(r'\b[0-9]+\.\s+\w+|\b[A-Za-z]+\s+[A-Za-z]+', text)
    return headers

def extract_text_chunk_faster(pdf_path, start_page=0, num_pages=5):
    """Extract a chunk of the PDF using PyMuPDF (fitz) for faster processing."""
    text_chunk = ""
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(start_page, min(start_page + num_pages, pdf_document.page_count)):
            page = pdf_document.load_page(page_num)
            text_chunk += page.get_text("text") + "\n"
        pdf_document.close()
        return text_chunk
    except Exception as e:
        logging.error(f"Error extracting text from PDF with PyMuPDF: {e}")
        return ""

async def async_extract_toc_llm(client, text_chunk):
    """Asynchronous LLM API request to speed up multiple requests."""
    logging.info(f"Sending request to OpenAI for TOC extraction...")

    prompt = (
        "You are analyzing a corporate handbook or business document. This document contains clearly defined sections, subsections, "
        "and page numbers in its Table of Contents (TOC). Your task is to extract all topics, subtopics, and corresponding "
        "page numbers. If page numbers are not provided, make a note. Please return the result in a structured JSON format, "
        "organized by section and subsection, with clear page numbers where possible."
        "\n\nDocument content:\n\n" + text_chunk
    )

    try:
        response = await client.chat.completions.create(
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

async def process_chunks_concurrently(client, chunks):
    """Process multiple chunks concurrently to speed up API requests."""
    tasks = [async_extract_toc_llm(client, chunk) for chunk in chunks]
    return await asyncio.gather(*tasks)

def save_toc_and_plan_to_json(data, output_file):
    """Save the extracted TOC or planning strategy into a JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"TOC and plan saved to JSON successfully at {output_file}.")
        print(f"TOC and plan saved to: {output_file}")
    except Exception as e:
        logging.error(f"Error saving JSON file: {e}")

def post_process_toc(toc_data):
    """Perform any necessary post-processing on the accumulated TOC data for consistency and cleanliness."""
    processed_toc = {}
    
    for section, content in toc_data.items():
        if isinstance(content, dict):
            processed_section = {
                "Title": section.title(),
                "Topics": content.get("Topics", [])
            }
        elif isinstance(content, list):
            processed_section = {
                "Title": section.title(),
                "Topics": content  # Assuming this list contains topic entries
            }
        else:
            logging.warning(f"Unexpected content structure for section {section}. Skipping post-processing.")
            continue
        
        # Add processed section back to the TOC
        processed_toc[processed_section["Title"]] = processed_section
    
    return processed_toc

async def main_async():
    pdf_path = r'C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\data\raw\Handbook.PDF'
    toc_output_file = os.path.join("data", "toc_output.json")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(toc_output_file), exist_ok=True)
    
    # Load OpenAI client
    client = load_openai_client()
    if not client:
        logging.error("OpenAI client could not be loaded. Exiting.")
        return
    
    # Extract TOC using LLM in chunks
    start_page = 0

    # Open the PDF to get the total number of pages for the progress bar
    total_pages = fitz.open(pdf_path).page_count
    chunk_size = dynamic_chunk_size(total_pages)
    accumulated_toc = {}

    chunks = []

    # Use tqdm to display progress bar
    with tqdm(total=total_pages, desc="Processing TOC", unit="page") as pbar:
        while start_page < total_pages:
            # Extract a chunk of the PDF
            text_chunk = extract_text_chunk_faster(pdf_path, start_page, chunk_size)
            if not text_chunk:
                break

            # Detect headers to guide splitting
            headers = detect_headers(text_chunk)
            logging.info(f"Detected headers: {headers}")

            chunks.append(text_chunk)

            # Move to the next chunk of pages and update progress bar
            start_page += chunk_size
            pbar.update(chunk_size)

    # Process chunks concurrently using async
    toc_results = await process_chunks_concurrently(client, chunks)

    for toc_chunk in toc_results:
        if toc_chunk:
            accumulated_toc.update(toc_chunk)

    if accumulated_toc:
        # Post-process the TOC to ensure consistency
        processed_toc = post_process_toc(accumulated_toc)
        
        # Save the accumulated TOC to a JSON file
        save_toc_and_plan_to_json(processed_toc, toc_output_file)
    else:
        logging.error("TOC extraction failed or incomplete.")

    logging.info("TOC extraction process completed.")

if __name__ == "__main__":
    asyncio.run(main_async())
