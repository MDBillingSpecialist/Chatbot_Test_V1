import os
import logging
import PyPDF2

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    logging.debug(f"Attempting to extract text from PDF: {pdf_path}")
    text = ""
    if not os.path.exists(pdf_path):
        logging.error(f"File not found: {pdf_path}")
        return text

    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
                logging.debug(f"Extracted text from page {page_num + 1}/{len(reader.pages)}")
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
    
    logging.info(f"Extracted {len(text)} characters from document.")
    return text

if __name__ == "__main__":
    file_path = 'Handbook.pdf'
    text = extract_text_from_pdf(file_path)
    if text:
        with open('output/full_text.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        logging.info("Text extracted and saved to 'output/full_text.txt'.")
