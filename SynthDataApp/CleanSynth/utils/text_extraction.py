
import PyPDF2
import logging

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        logging.debug(f"Attempting to open PDF file: {pdf_path}")
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)  # This line uses PyPDF2, so it must be imported
            for page_num, page in enumerate(reader.pages):
                extracted_text = page.extract_text()
                logging.debug(f"Extracted text from page {page_num + 1}")
                if extracted_text:
                    text += extracted_text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
    logging.info(f"Extracted {len(text)} characters from document.")
    return text
