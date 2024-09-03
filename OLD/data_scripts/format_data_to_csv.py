import os
import logging
from config import CLEANED_DATA_PATH, FORMATTED_CSV_PATH

logging.basicConfig(level=logging.INFO)

def format_data_to_csv(cleaned_data, output_path):
    paragraphs = cleaned_data.split('\n\n')
    with open(output_path, 'w', encoding='utf-8') as file:
        for para in paragraphs:
            file.write(f'"{para.strip()}"\n')

if __name__ == "__main__":
    try:
        with open(CLEANED_DATA_PATH, 'r', encoding='utf-8') as f:
            cleaned_data = f.read()
        format_data_to_csv(cleaned_data, FORMATTED_CSV_PATH)
        logging.info(f"Data formatting to CSV complete. Data saved to {FORMATTED_CSV_PATH}.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
