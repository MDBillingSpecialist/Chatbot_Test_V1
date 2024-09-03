import os
import logging
import json
import PyPDF2
import re
import hashlib
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        logging.debug(f"Attempting to open PDF file: {pdf_path}")
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                extracted_text = page.extract_text()
                logging.debug(f"Extracted text from page {page_num + 1}")
                if extracted_text:
                    text += extracted_text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
    logging.info(f"Extracted {len(text)} characters from document.")
    return text

def chunk_document(text, chunk_size=10000):
    """Chunk the document into manageable pieces without overlap."""
    logging.debug("Starting the document chunking process.")
    chunks = []
    text_length = len(text)
    i = 0

    while i < text_length:
        end = min(i + chunk_size, text_length)
        chunk = text[i:end]
        
        # Try to avoid cutting off in the middle of a paragraph
        if end < text_length and text[end] != '\n':
            while end < text_length and text[end] != '\n':
                end += 1
            chunk = text[i:end]
        
        chunk_hash = hashlib.md5(chunk.encode('utf-8')).hexdigest()
        if not chunks or chunk_hash != hashlib.md5(chunks[-1].encode('utf-8')).hexdigest():
            chunks.append(chunk)
            logging.debug(f"Chunk {len(chunks)} added. Length: {len(chunk)} characters.")
        i = end  # Move to the next chunk without overlap

    logging.info(f"Document split into {len(chunks)} chunks without overlap.")
    return chunks

def use_llm_for_toc_as_json(client, text_chunk):
    """Use GPT-4o mini to extract the TOC and return it as a JSON-like structure."""
    logging.info("Sending request to OpenAI to analyze TOC and return it as JSON...")

    prompt = (
        "You are analyzing a corporate handbook. The following text contains the Table of Contents (TOC). "
        "Please return the TOC as a structured JSON object, where each section title is a key and its corresponding page number is the value. "
        "If there are subsections, include them as nested JSON objects. The text to analyze is below:\n\n"
        f"{text_chunk}"
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
        toc_json = response.choices[0].message.content
        logging.info("TOC JSON extraction completed.")
        logging.debug(f"Extracted TOC JSON: {toc_json}")
        return toc_json
    except Exception as e:
        logging.error(f"Error during TOC JSON extraction: {e}")
        return None

def save_segment_to_json(section_title, segment, output_file):
    """Save a single segmented text to a JSON file incrementally."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load existing content
    try:
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logging.debug(f"Loaded existing JSON content: {json.dumps(data, indent=4)[:500]}...")  # Log the start of the existing JSON content
        else:
            data = {}
            logging.debug("No existing JSON content found. Starting fresh.")
    except json.JSONDecodeError as e:
        logging.error(f"Error loading JSON file: {e}")
        data = {}

    # Update the content with the new segment
    if not segment.strip():
        logging.warning(f"Skipping empty segment for section: {section_title}")
        return

    data[section_title] = segment
    logging.debug(f"Updated JSON with section: {section_title}")

    # Save the updated content back to the JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info(f"Segment '{section_title}' saved to {output_file}")
        logging.debug(f"Current JSON content: {json.dumps(data, f, indent=4)[:500]}...")  # Log the start of the updated JSON content
    except Exception as e:
        logging.error(f"Error saving segment '{section_title}': {e}")

def parse_segmented_text(segmented_text, toc_entries):
    """Parse the segmented text into sections based on TOC entries."""
    segments = {}
    current_section = None
    current_content = []

    logging.debug(f"Parsing segmented text...")
    logging.debug(f"TOC Entries: {toc_entries}")  # Log TOC Entries being used

    for line in segmented_text.splitlines():
        line = line.strip()
        logging.debug(f"Processing line: {line[:100]}")  # Log the current line being processed
        if any(line.startswith(f"# {entry}") for entry in toc_entries):  # Assuming the LLM uses Markdown headings
            if current_section:
                segments[current_section] = "\n".join(current_content)
                logging.debug(f"Segment parsed for section: {current_section}")
            current_section = line
            current_content = []
        elif current_section:
            current_content.append(line)

    if current_section:  # Don't forget to save the last section
        segments[current_section] = "\n".join(current_content)
        logging.debug(f"Segment parsed for section: {current_section}")

    logging.debug(f"Parsed segments: {segments.keys()}")  # Log the parsed segment titles
    return segments

def segment_based_on_toc_and_subtopics(text, toc_entries, output_file):
    """Segment the document text based on TOC entries and save each segment immediately."""
    chunks = chunk_document(text)
    found_sections = {}

    for i, chunk in enumerate(chunks):
        logging.debug(f"Processing chunk {i + 1}/{len(chunks)}")
        remaining_toc = [entry for entry in toc_entries if entry not in found_sections]
        if not remaining_toc:
            logging.info("All sections have been processed.")
            break

        segmented_text = use_llm_for_segmentation(client, chunk, remaining_toc)
        if segmented_text:
            parsed_segments = parse_segmented_text(segmented_text, remaining_toc)
            for section, content in parsed_segments.items():
                save_segment_to_json(section, content, output_file)
                found_sections[section] = True
        else:
            logging.warning(f"Segmentation failed for chunk {i + 1}")

    logging.info("Document segmentation process completed.")

def process_document_for_segmentation(file_path):
    """Process the document for chunking and segmentation using model-driven TOC parsing."""
    logging.info(f"Starting document processing for: {file_path}")
    text = extract_text_from_pdf(file_path)
    
    chunks = chunk_document(text)
    combined_toc_text = " ".join(chunks[:3])  # Combine the first few chunks for TOC extraction
    logging.debug(f"Checking combined chunks for TOC...")
    toc_json_str = use_llm_for_toc_as_json(client, combined_toc_text)

    if toc_json_str:
        try:
            toc_entries = json.loads(toc_json_str)  # Parse the JSON string into a dictionary
            logging.info("TOC JSON parsed successfully.")
            segment_based_on_toc_and_subtopics(text, toc_entries, 'output/segments.json')
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse TOC JSON: {e}")
            logging.warning("Falling back on basic segmentation.")
            for i, chunk in enumerate(chunks):
                logging.debug(f"Processing unsegmented chunk {i + 1}/{len(chunks)}")
                save_segment_to_json(f"Unsegmented Chunk {i + 1}", chunk, 'output/segments.json')
    else:
        logging.warning("No TOC JSON detected. Falling back on basic segmentation.")
        for i, chunk in enumerate(chunks):
            logging.debug(f"Processing unsegmented chunk {i + 1}/{len(chunks)}")
            save_segment_to_json(f"Unsegmented Chunk {i + 1}", chunk, 'output/segments.json')

    logging.info("Segmentation process completed.")

# Example Usage
if __name__ == "__main__":
    file_path = 'Handbook.PDF'
    process_document_for_segmentation(file_path)
    print("Document segmentation complete.")
