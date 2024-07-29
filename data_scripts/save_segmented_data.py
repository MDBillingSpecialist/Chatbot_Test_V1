import os
import re
import logging
from config import SEGMENTED_DATA_PATH, SEGMENTED_DIR

logging.basicConfig(level=logging.INFO)

def save_segmented_data(segmented_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for heading, content in segmented_data.items():
        heading_clean = re.sub(r'[^\w\s]', '', heading).replace(' ', '_')
        with open(os.path.join(output_dir, f'{heading_clean}.txt'), 'w', encoding='utf-8') as file:
            file.write(content)

if __name__ == "__main__":
    try:
        segmented_data = {}
        with open(SEGMENTED_DATA_PATH, 'r', encoding='utf-8') as f:
            current_heading = None
            current_content = []
            for line in f:
                if re.match(r'^[A-Z][A-Z ]*$', line.strip()):
                    if current_heading:
                        segmented_data[current_heading] = '\n'.join(current_content)
                    current_heading = line.strip()
                    current_content = []
                else:
                    current_content.append(line.strip())
            if current_heading:
                segmented_data[current_heading] = '\n'.join(current_content)
        save_segmented_data(segmented_data, SEGMENTED_DIR)
        logging.info(f"Segmented data saved to individual files in '{SEGMENTED_DIR}' directory.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
