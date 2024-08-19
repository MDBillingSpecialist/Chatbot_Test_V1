import PyPDF2
import os
import re
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
    except Exception as e:
        logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
    return text

def extract_toc(text):
    """Try to locate the Table of Contents by looking for common TOC patterns."""
    toc_start = None
    toc_end = None
    lines = text.splitlines()

    # Basic keyword search for TOC start and end
    for i, line in enumerate(lines):
        if "Table of Contents" in line or "Contents" in line:
            toc_start = i
        if toc_start and re.match(r'^\d+\s+', line):  # Assuming TOC ends when content starts
            toc_end = i
            break

    if toc_start is not None and toc_end is not None:
        toc_lines = lines[toc_start:toc_end]
        logging.info(f"TOC found from line {toc_start} to {toc_end}")
    else:
        logging.warning("No TOC found in the document.")

    return toc_lines if toc_start is not None else []


def extract_headers(text):
    """Extract headers from the document using a broader header pattern."""
    headers = []
    # Enhanced header pattern, consider adjusting based on document's structure
    header_pattern = re.compile(r'^(?:\d+\.\s)?[A-Z][\w\s\-]+(?:\s\d+)?$')
    lines = text.splitlines()

    for line in lines:
        if header_pattern.match(line.strip()):
            headers.append(line.strip())

    if not headers:
        logging.warning("No headers found.")
    else:
        logging.info(f"Extracted {len(headers)} headers.")
    return headers

def segment_text(text, toc, headers):
    """Segment text based on TOC and headers."""
    segments = []
    current_segment = ""
    lines = text.splitlines()

    # Set thresholds
    toc_threshold = 2
    header_threshold = 2

    # Start segmenting based on TOC and headers
    for line in lines:
        stripped_line = line.strip()
        
        if stripped_line in toc:
            if len(current_segment) > 0:
                segments.append(current_segment.strip())
                current_segment = ""
            current_segment += stripped_line + "\n"
        elif stripped_line in headers:
            if len(current_segment) > 0:
                segments.append(current_segment.strip())
                current_segment = ""
            current_segment += stripped_line + "\n"
        else:
            current_segment += stripped_line + "\n"

    if len(current_segment) > 0:
        segments.append(current_segment.strip())

    if len(segments) <= toc_threshold or len(segments) <= header_threshold:
        logging.error("No segments created. Check TOC and header extraction.")
        return []

    logging.info(f"Segmented into {len(segments)} segments.")
    return segments

def save_segments_to_json(segments, output_file):
    """Save segments to a JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=4)
        logging.info(f"Segments saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving segments to JSON: {e}")

def process_document(pdf_path):
    """Main processing function."""
    logging.info(f"Starting processing for document: {pdf_path}")

    # Extract text from the document
    text = extract_text_from_pdf(pdf_path)
    logging.info(f"Extracted {len(text)} characters from document.")

    # Extract Table of Contents and headers
    toc = extract_toc(text)
    headers = extract_headers(text)

    if not toc and not headers:
        logging.error("No valid TOC or headers found. Aborting segmentation.")
        return

    # Segment text
    segments = segment_text(text, toc, headers)

    if not segments:
        logging.error("No segments created. Aborting process.")
        return

    # Save segments to JSON
    output_file = os.path.join(os.path.dirname(pdf_path), 'segments.json')
    save_segments_to_json(segments, output_file)

if __name__ == "__main__":
    # Test with the provided Handbook.PDF
    process_document('Handbook.PDF')
