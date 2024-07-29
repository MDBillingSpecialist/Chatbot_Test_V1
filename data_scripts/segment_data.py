import re
import os
import logging
from config import CLEANED_DATA_PATH, SEGMENTED_DATA_PATH

logging.basicConfig(level=logging.INFO)

def segment_data(cleaned_data):
    sections = re.split(r'(\n[A-Z][A-Z ]*\n)', cleaned_data)
    segmented_data = {}

    if sections[0].strip() == '':
        sections = sections[1:]

    for i in range(0, len(sections), 2):
        heading = sections[i].strip()
        content = sections[i + 1].strip() if i + 1 < len(sections) else ''
        if heading and content:
            segmented_data[heading] = content

    return segmented_data

if __name__ == "__main__":
    try:
        with open(CLEANED_DATA_PATH, 'r', encoding='utf-8') as f:
            cleaned_data = f.read()
        segmented_data = segment_data(cleaned_data)
        with open(SEGMENTED_DATA_PATH, 'w', encoding='utf-8') as f:
            for heading, content in segmented_data.items():
                f.write(f'{heading}\n{content}\n\n')
        logging.info(f"Data segmentation complete. Segmented data saved to {SEGMENTED_DATA_PATH}.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
