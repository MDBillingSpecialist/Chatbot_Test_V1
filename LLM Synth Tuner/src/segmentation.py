import re
import pdfplumber
import logging
import json
import os
from openai import OpenAI
from typing import Dict, Union
import argparse
from tqdm import tqdm  # Import tqdm for the progress bar

# Setup logging configuration
logging.basicConfig(
    filename='pdf_segmentation_rag.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def load_openai_client():
    """Load OpenAI client with API key from environment."""
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OpenAI API key not found. Please set it in the .env file.")
        return None
    return OpenAI(api_key=api_key)

def create_toc_patterns(toc_structure: Dict[str, Union[str, Dict[str, str]]]) -> Dict[str, re.Pattern]:
    """Creates a dictionary of compiled regex patterns for all TOC entries."""
    patterns = {}
    for section, subsections in toc_structure.items():
        pattern = rf"^\s*(?:\d+\s*)?{re.escape(section)}(?:[:\s]|$)"
        patterns[section] = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        if isinstance(subsections, dict):
            for subsection in subsections:
                sub_pattern = rf"^\s*(?:\d+\.\d+\s*)?{re.escape(subsection)}(?:[:\s]|$)"
                patterns[subsection] = re.compile(sub_pattern, re.IGNORECASE | re.MULTILINE)
    return patterns

def fuzzy_match_title_with_llm(client, line: str, toc_structure: Dict[str, Union[str, Dict[str, str]]]) -> Union[str, None]:
    """Use LLM to perform a smarter matching on the line against the TOC structure."""
    prompt = f"Match the following line to the most relevant section in the Table of Contents:\n\nTOC: {json.dumps(toc_structure)}\n\nLine: {line}\n\nPlease provide the most appropriate section name."
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Specify the mini model here
            messages=[{"role": "system", "content": "You are an expert in document segmentation."},
                    {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=100
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logging.error(f"Error in LLM-based TOC matching: {e}")
        return None

def segment_pdf_with_llm(pdf_path: str, toc_structure: Dict[str, Union[str, Dict[str, str]]], client) -> Dict[str, Dict]:
    """Extract and segment the text from the PDF based on the provided Table of Contents structure using LLM assistance."""
    logging.info("Starting PDF segmentation process with LLM assistance.")
    text_segments = {}
    current_title = None
    current_segment = []
    
    try:
        # Open the PDF using a context manager to ensure proper handling of the file object
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logging.info(f"Opened PDF file: {pdf_path}. Total pages: {total_pages}")
            
            # Set up progress bar
            with tqdm(total=total_pages, desc="Processing PDF", unit="page") as pbar:
                for page_num in range(total_pages):
                    page = pdf.pages[page_num]
                    logging.info(f"Processing page {page_num + 1} of {total_pages}")
                    text = page.extract_text()
                    if not text:
                        logging.warning(f"No text found on page {page_num + 1}.")
                        continue

                    lines = text.split('\n')
                    
                    for line in lines:
                        # Use LLM to match the line with the most relevant TOC entry
                        title = fuzzy_match_title_with_llm(client, line, toc_structure)
                        if title:
                            if current_title and current_segment:
                                text_segments[current_title] = {
                                    "text": "\n".join(current_segment).strip(),
                                    "start_page": page_num + 1,
                                    "end_page": page_num + 1
                                }
                                current_segment = []
                            current_title = title
                            logging.debug(f"Title detected by LLM: '{title}' - Creating new segment for {current_title}.")
                        elif current_title:
                            current_segment.append(line.strip())
                    
                    logging.info(f"Finished processing page {page_num + 1}.")
                    
                    # Update progress bar
                    pbar.update(1)
            
            # Add the final segment if any
            if current_title and current_segment:
                text_segments[current_title] = {
                    "text": "\n".join(current_segment).strip(),
                    "start_page": page_num + 1,
                    "end_page": page_num + 1
                }

    except Exception as e:
        logging.error(f"Error during PDF processing: {e}")
        return {}

    logging.info("PDF segmentation with LLM assistance completed successfully.")
    return text_segments


def save_segments_with_metadata_to_json(segments: Dict[str, Dict], output_file: str):
    """Save the segmented text and metadata (start_page, end_page) into a JSON file."""
    if not segments:
        logging.warning("No segments to save. Exiting save function.")
        return

    logging.info(f"Saving segments with metadata to JSON file: {output_file}.")
    try:
        with open(output_file, 'w') as f:
            json.dump(segments, f, indent=4)
        logging.info("Segments with metadata saved to JSON successfully.")
    except Exception as e:
        logging.error(f"Error saving JSON file: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="PDF Segmentation Tool with LLM for RAG and Citation Infrastructure")
    parser.add_argument("--config", default="config.json", nargs="?", help="Path to the configuration file")
    return parser.parse_args()

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config file: {e}")
        raise e

def main():
    args = parse_arguments()

    # Load the configuration from the specified path or default to 'config.json'
    config_path = args.config or "config.json"
    config = load_config(config_path)

    pdf_path = config.get("pdf_path")
    toc_json_path = config.get("toc_json_path")
    output_folder = config.get("output_folder")
    output_file = config.get("output_file")

    if not os.path.exists(toc_json_path):
        logging.error(f"TOC JSON file not found at {toc_json_path}")
        print(f"Error: TOC JSON file not found at {toc_json_path}")
        return
    
    # Load the TOC structure from the JSON file
    with open(toc_json_path, 'r') as f:
        toc_structure = json.load(f)
    
    # Load OpenAI client
    client = load_openai_client()
    if not client:
        logging.error("Failed to initialize OpenAI client.")
        return
    
    # Extract and segment the text using the LLM for better matching
    segments = segment_pdf_with_llm(pdf_path, toc_structure, client)
    
    if segments:
        # Save the segmented text with metadata to a JSON file
        os.makedirs(output_folder, exist_ok=True)
        json_output_file = os.path.join(output_folder, args.output)
        save_segments_with_metadata_to_json(segments, json_output_file)
        print(f"Segmentation completed. Output saved to: {json_output_file}")
    else:
        logging.warning("No segments to save. The segmentation process may have failed.")
        print("Error: No segments were extracted. Please check the log file for details.")

    logging.info("Script execution completed.")

if __name__ == "__main__":
    main()
