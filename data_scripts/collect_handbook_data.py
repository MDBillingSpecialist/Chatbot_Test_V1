from PyPDF2 import PdfReader
import os
import logging
from config import PDF_PATH, HANDBOOK_DATA_PATH

logging.basicConfig(level=logging.INFO)

def collect_handbook_data(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            num_pages = len(reader.pages)
            data = ''
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                data += page.extract_text()
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    data = collect_handbook_data(PDF_PATH)
    if data:
        with open(HANDBOOK_DATA_PATH, 'w', encoding='utf-8') as f:
            f.write(data)
        logging.info(f"Data collection complete. Data saved to {HANDBOOK_DATA_PATH}.")
