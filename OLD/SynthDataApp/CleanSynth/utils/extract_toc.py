import re
import pdfplumber
import logging
import json

# Setup logging configuration
logging.basicConfig(
    filename='toc_extraction.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def is_probably_toc_page(text):
    """Heuristic to determine if a page is likely part of the TOC."""
    lines = text.split('\n')
    toc_indicators = ['Table of Contents', '.....', 'Chapter', 'Page']
    return any(any(indicator in line for indicator in toc_indicators) for line in lines)

def clean_toc_text(toc_text):
    """Clean up and format the extracted TOC text."""
    # Split the text by lines
    lines = toc_text.split('\n')
    cleaned_lines = []

    for line in lines:
        # Remove unnecessary whitespace
        cleaned_line = line.strip()
        
        # Skip empty lines
        if not cleaned_line:
            continue
        
        # Remove any extraneous dots or other TOC formatting characters
        cleaned_line = re.sub(r'\.{2,}', '', cleaned_line)
        
        # Add the cleaned line to the list
        cleaned_lines.append(cleaned_line)

    return '\n'.join(cleaned_lines)

def extract_toc(pdf_path):
    """Extract the Table of Contents from the PDF."""
    logging.info("Starting TOC extraction process.")
    toc_text = ""
    toc_pages = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    logging.warning(f"No text found on page {page_num}.")
                    continue

                if is_probably_toc_page(text):
                    toc_text += text + "\n"
                    toc_pages.append(page_num)
                    logging.info(f"Identified TOC page: {page_num}")
                else:
                    # Stop extracting once TOC is fully captured
                    if toc_text:
                        break

    except Exception as e:
        logging.error(f"Error during TOC extraction: {e}")
        return None, []

    # Clean up and format the TOC text
    toc_text = clean_toc_text(toc_text)
    
    logging.info("TOC extraction completed successfully.")
    return toc_text.strip(), toc_pages

def save_toc(toc_text, output_file):
    """Save the TOC text into a JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump({"toc": toc_text}, f, indent=4)
        logging.info("TOC saved to JSON successfully.")
    except Exception as e:
        logging.error(f"Error saving TOC JSON file: {e}")

def main():
    pdf_path = r'C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\SynthDataApp\CleanSynth\utils\Handbook.PDF'
    toc_output_file = 'toc_output.json'
    
    toc_text, toc_pages = extract_toc(pdf_path)
    
    if toc_text:
        save_toc(toc_text, toc_output_file)
    else:
        logging.warning("No TOC extracted. Please check the PDF and script.")
    
    logging.info("TOC extraction script completed.")

if __name__ == "__main__":
    main()
