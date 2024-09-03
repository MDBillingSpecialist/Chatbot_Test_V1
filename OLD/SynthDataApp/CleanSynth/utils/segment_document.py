import re
import pdfplumber
import logging
import json
import os

# Setup logging configuration
logging.basicConfig(
    filename='document_segmentation.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def create_toc_patterns(toc_text):
    """Creates a dictionary of compiled regex patterns for all TOC entries."""
    toc_lines = toc_text.split('\n')
    patterns = {}
    for line in toc_lines:
        # Extract the section title without page numbers
        title = re.split(r'\s\d+$', line.strip())[0]
        patterns[title] = re.compile(f"^{re.escape(title)}(?:\\s|$)", re.IGNORECASE)
    return patterns

def segment_pdf(pdf_path, toc_text, toc_pages):
    """Extract and segment the text from the PDF based on the TOC."""
    logging.info("Starting PDF segmentation process.")
    text_segments = {}
    current_title = None
    current_segment = []
    
    toc_patterns = create_toc_patterns(toc_text)
    toc_page_set = set(toc_pages)
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                if page_num in toc_page_set:
                    continue

                text = page.extract_text()
                if not text:
                    logging.warning(f"No text found on page {page_num}.")
                    continue

                logging.info(f"Processing page {page_num}.")
                lines = text.split('\n')
                
                for line in lines:
                    for title, pattern in toc_patterns.items():
                        if pattern.match(line):
                            if current_title and current_segment:
                                text_segments[current_title] = " ".join(current_segment).strip()
                                current_segment = []
                            current_title = title
                            logging.debug(f"Title detected: '{line.strip()}' - Creating new segment for {current_title}.")
                            break
                    
                    if current_title:
                        current_segment.append(line.strip())
                
                logging.info(f"Finished processing page {page_num}.")
            
            if current_title and current_segment:
                text_segments[current_title] = " ".join(current_segment).strip()

    except Exception as e:
        logging.error(f"Error during PDF processing: {e}")
        return {}

    logging.info("PDF segmentation completed successfully.")
    return text_segments

def save_segments_to_json(segments, output_file):
    """Save the segmented text into a clean JSON file."""
    if not segments:
        logging.warning("No segments to save. Exiting save function.")
        return

    logging.info(f"Saving segments to JSON file: {output_file}.")
    try:
        with open(output_file, 'w') as f:
            json.dump(segments, f, indent=4)
        logging.info("Segments saved to JSON successfully.")
    except Exception as e:
        logging.error(f"Error saving JSON file: {e}")

def main():
    pdf_path = r'C:\path\to\your\document.pdf'
    toc_json_file = 'toc_output.json'
    output_folder = './segmented_output'
    json_output_file = os.path.join(output_folder, 'segmented_output.json')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logging.info(f"Created output directory: {output_folder}")

    with open(toc_json_file, 'r') as f:
        toc_data = json.load(f)
    
    toc_text = toc_data['toc']
    toc_pages = toc_data.get('toc_pages', [])

    segments = segment_pdf(pdf_path, toc_text, toc_pages)
    
    if segments:
        save_segments_to_json(segments, json_output_file)
    else:
        logging.warning("No segments to save. The segmentation process may have failed.")

    logging.info("Script execution completed.")

if __name__ == "__main__":
    main()
