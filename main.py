import os
import logging
from data_scripts.collect_handbook_data import collect_handbook_data
from data_scripts.clean_data import clean_data
from data_scripts.format_data_to_csv import format_data_to_csv
from data_scripts.segment_data import segment_data
from data_scripts.save_segmented_data import save_segmented_data
from config import PDF_PATH, HANDBOOK_DATA_PATH, CLEANED_DATA_PATH, FORMATTED_CSV_PATH, SEGMENTED_DATA_PATH, SEGMENTED_DIR

logging.basicConfig(level=logging.INFO)

def main():
    try:
        # Collect data
        handbook_data = collect_handbook_data(PDF_PATH)
        if handbook_data:
            with open(HANDBOOK_DATA_PATH, 'w', encoding='utf-8') as f:
                f.write(handbook_data)
            logging.info("Data collection complete.")
        
        # Clean data
        with open(HANDBOOK_DATA_PATH, 'r', encoding='utf-8') as f:
            handbook_data = f.read()
        cleaned_data = clean_data(handbook_data)
        with open(CLEANED_DATA_PATH, 'w', encoding='utf-8') as f:
            f.write(cleaned_data)
        logging.info("Data cleaning complete.")
        
        # Format data to CSV
        format_data_to_csv(cleaned_data, FORMATTED_CSV_PATH)
        logging.info("Data formatting to CSV complete.")
        
        # Segment data
        segmented_data = segment_data(cleaned_data)
        with open(SEGMENTED_DATA_PATH, 'w', encoding='utf-8') as f:
            for heading, content in segmented_data.items():
                f.write(f'{heading}\n{content}\n\n')
        logging.info("Data segmentation complete.")
        
        # Save segmented data
        save_segmented_data(segmented_data, SEGMENTED_DIR)
        logging.info("Segmented data saved to individual files.")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
