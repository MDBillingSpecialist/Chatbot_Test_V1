import re
import pdfplumber
import logging
import json
import os
import yaml
from typing import Dict, Union, List, Tuple
from tqdm import tqdm

# Setup logging configuration
logging.basicConfig(
    filename='pdf_segmentation_toc.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Optional: Logic-based max paragraph length for further splitting
MAX_PARAGRAPHS_PER_SEGMENT = 10  # Split if a segment exceeds this number of paragraphs

def load_toc(toc_json_path: str) -> Dict[str, Union[str, Dict[str, str]]]:
    """Load the Table of Contents (TOC) from a JSON file."""
    with open(toc_json_path, 'r') as f:
        toc_structure = json.load(f)
    return toc_structure

def load_config(config_path: str) -> Dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_toc_patterns(toc_structure: Dict[str, Union[str, Dict[str, str]]]) -> List[Tuple[str, re.Pattern]]:
    """Creates a list of tuples containing subsection titles and compiled regex patterns."""
    patterns = []
    for section, subsections in toc_structure["Table of Contents"].items():
        if isinstance(subsections, dict):
            for subsection in subsections:
                pattern = rf"^\s*{re.escape(subsection)}[\s:]*$"
                patterns.append((subsection, re.compile(pattern, re.IGNORECASE | re.MULTILINE)))
    
    return patterns

def split_logically_by_paragraphs(text: str, title: str) -> List[Dict]:
    """Logically split large sections based on paragraphs."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]  # Split by double newlines for paragraphs
    segments = []
    
    # Split the paragraphs into manageable chunks
    for i in range(0, len(paragraphs), MAX_PARAGRAPHS_PER_SEGMENT):
        chunk_paragraphs = paragraphs[i:i + MAX_PARAGRAPHS_PER_SEGMENT]
        chunk_text = "\n\n".join(chunk_paragraphs)
        chunk_title = f"{title} - Part {i // MAX_PARAGRAPHS_PER_SEGMENT + 1}"
        segments.append({"title": chunk_title, "text": chunk_text})
    
    return segments

def segment_pdf_using_toc(pdf_path: str, toc_patterns: List[Tuple[str, re.Pattern]]) -> Dict[str, Dict]:
    """Extract and segment the text from the PDF based on TOC subsection patterns."""
    logging.info("Starting PDF segmentation process using TOC subsections.")
    text_segments = {}
    current_title = None
    current_segment = []
    start_page = None
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logging.info(f"Opened PDF file: {pdf_path}. Total pages: {total_pages}")
            
            # Set up progress bar
            with tqdm(total=total_pages, desc="Processing PDF", unit="page") as pbar:
                for page_num in range(total_pages):
                    page = pdf.pages[page_num]
                    text = page.extract_text()

                    # Handle blank or unrecognized text extraction issues
                    if not text:
                        logging.warning(f"No text found on page {page_num + 1}. Attempting fallback extraction method.")
                        text = page.extract_words()  # Fallback extraction
                        if not text:
                            logging.error(f"Failed to extract text from page {page_num + 1}. Skipping.")
                            continue

                    lines = text.split('\n') if isinstance(text, str) else [t['text'] for t in text]

                    for line in lines:
                        matched = False
                        for title, pattern in toc_patterns:
                            if pattern.match(line):
                                # Save the current segment if it exists
                                if current_title and current_segment:
                                    segment_text = "\n".join(current_segment).strip()
                                    if len(segment_text.split('\n\n')) > MAX_PARAGRAPHS_PER_SEGMENT:
                                        # Logically split large sections by paragraphs
                                        for chunk in split_logically_by_paragraphs(segment_text, current_title):
                                            text_segments[chunk["title"]] = {
                                                "text": chunk["text"],
                                                "start_page": start_page,
                                                "end_page": page_num
                                            }
                                    else:
                                        text_segments[current_title] = {
                                            "text": segment_text,
                                            "start_page": start_page,
                                            "end_page": page_num
                                        }
                                    current_segment = []
                                
                                current_title = title
                                start_page = page_num + 1
                                logging.debug(f"Subsection detected: '{title}' - Creating new segment.")
                                matched = True
                                break
                        
                        if current_title and not matched:
                            current_segment.append(line.strip())
                    
                    logging.info(f"Finished processing page {page_num + 1}.")
                    pbar.update(1)
            
            # Add the final segment if any
            if current_title and current_segment:
                segment_text = "\n".join(current_segment).strip()
                if len(segment_text.split('\n\n')) > MAX_PARAGRAPHS_PER_SEGMENT:
                    for chunk in split_logically_by_paragraphs(segment_text, current_title):
                        text_segments[chunk["title"]] = {
                            "text": chunk["text"],
                            "start_page": start_page,
                            "end_page": total_pages
                        }
                else:
                    text_segments[current_title] = {
                        "text": segment_text,
                        "start_page": start_page,
                        "end_page": total_pages
                    }

    except Exception as e:
        logging.error(f"Error during PDF processing: {e}")
        return {}

    logging.info("PDF segmentation using TOC subsections completed successfully.")
    return text_segments

def save_segments_to_json(segments: Dict[str, Dict], output_file: str):
    """Save the segmented text into a JSON file."""
    if not segments:
        logging.warning("No segments to save. Exiting save function. Outputting first few lines for debug.")
        try:
            with pdfplumber.open(config['pdf_path']) as pdf:
                for page in pdf.pages[:3]:
                    logging.info(f"First few lines of the document for debug: {page.extract_text()[:500]}")
        except Exception as e:
            logging.error(f"Error during fallback debug extraction: {e}")
        return

    output_folder = os.path.dirname(output_file)
    os.makedirs(output_folder, exist_ok=True)

    logging.info(f"Saving segments to JSON file: {output_file}.")
    try:
        with open(output_file, 'w') as f:
            json.dump(segments, f, indent=4)
        logging.info("Segments saved to JSON successfully.")
    except Exception as e:
        logging.error(f"Error saving JSON file: {e}")

def main():
    config_path = r'C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\src\config.yaml'  # Correct path to config YAML
    config = load_config(config_path)

    toc_json_path = config['toc_json_path']
    pdf_path = config['pdf_path']
    
    # Update the output folder and filename
    output_folder = r'C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\data\segmented_output'
    output_file = os.path.join(output_folder, 'segmented_output.json')

    toc_structure = load_toc(toc_json_path)
    toc_patterns = create_toc_patterns(toc_structure)
    segments = segment_pdf_using_toc(pdf_path, toc_patterns)

    if segments:
        save_segments_to_json(segments, output_file)
        print(f"Segmentation completed. Output saved to: {output_file}")
    else:
        print("Error: No segments were extracted. Please check the log file for details.")


if __name__ == "__main__":
    main()
