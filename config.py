import os

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, 'Handbook.PDF')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data_output')
HANDBOOK_DATA_PATH = os.path.join(OUTPUT_DIR, 'handbook_data.txt')
CLEANED_DATA_PATH = os.path.join(OUTPUT_DIR, 'cleaned_handbook_data.txt')
FORMATTED_CSV_PATH = os.path.join(OUTPUT_DIR, 'formatted_handbook.csv')
SEGMENTED_DATA_PATH = os.path.join(OUTPUT_DIR, 'segmented_data.txt')
SEGMENTED_DIR = os.path.join(OUTPUT_DIR, 'segmented_data')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SEGMENTED_DIR, exist_ok=True)
