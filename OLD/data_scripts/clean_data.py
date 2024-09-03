import re
import os
import logging
from config import HANDBOOK_DATA_PATH, CLEANED_DATA_PATH

logging.basicConfig(level=logging.INFO)

def clean_data(data):
    cleaned_data = re.sub(r'\([^)]*\)', '', data)
    cleaned_data = re.sub(r'\n{2,}', '\n\n', cleaned_data).strip()
    return cleaned_data

if __name__ == "__main__":
    try:
        with open(HANDBOOK_DATA_PATH, 'r', encoding='utf-8') as f:
            data = f.read()
        cleaned_data = clean_data(data)
        with open(CLEANED_DATA_PATH, 'w', encoding='utf-8') as f:
            f.write(cleaned_data)
        logging.info(f"Data cleaning complete. Cleaned data saved to {CLEANED_DATA_PATH}.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
