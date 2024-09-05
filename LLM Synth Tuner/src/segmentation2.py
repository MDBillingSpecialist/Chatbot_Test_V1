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
            
            with tqdm(total=total_pages, desc="Processing PDF", unit="page") as pbar:
                for page_num in range(total_pages):
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    if not text:
                        logging.warning(f"No text found on page {page_num + 1}.")
                        continue

                    lines = text.split('\n')
                    
                    for line in lines:
                        matched = False
                        for title, pattern in toc_patterns:
                            if pattern.match(line):
                                if current_title and current_segment:
                                    text_segments[current_title] = {
                                        "text": "\n".join(current_segment).strip(),
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
            
            if current_title and current_segment:
                text_segments[current_title] = {
                    "text": "\n".join(current_segment).strip(),
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
        logging.warning("No segments to save. Exiting save function.")
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
    config_path = r'C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\src\config.yaml'
    config = load_config(config_path)

    toc_json_path = config['toc_json_path']
    pdf_path = config['pdf_path']
    output_file = os.path.join(config['output_folder'], config['output_file'])

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