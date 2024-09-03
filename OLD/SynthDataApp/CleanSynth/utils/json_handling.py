import json
import logging

def save_json(filename, data):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info(f"Data successfully saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving data to {filename}: {e}")

def load_json(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Data successfully loaded from {filename}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {filename}: {e}")
        return None
