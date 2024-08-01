import os
import re
import json
import logging
from pdfminer.high_level import extract_text
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    try:
        text = extract_text(pdf_path)
        if not text.strip():
            raise ValueError("Extracted text is empty. Check the PDF content.")
        logging.info(f"Text successfully extracted from {pdf_path}")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return None

# Function to split the handbook into sections based on headings
def split_into_sections(text):
    try:
        sections = {}
        lines = text.splitlines()
        current_section = None

        for line in lines:
            if re.match(r'^[A-Z][A-Z\s]+$', line.strip()):  # Detect headings
                current_section = line.strip()
                sections[current_section] = ""
            elif current_section:
                sections[current_section] += line.strip() + " "

        if not sections:
            raise ValueError("No sections identified. Check the text formatting.")
        
        logging.info(f"Sections successfully identified: {list(sections.keys())}")
        return sections
    except Exception as e:
        logging.error(f"Error splitting text into sections: {e}")
        return {}

# Function to create question-answer pairs from sections
def create_qa_pairs(sections):
    try:
        qa_pairs = []
        
        for section, content in sections.items():
            logging.debug(f"Processing Section: {section}")
            logging.debug(f"Content Preview: {content[:200]}...")  # Preview first 200 characters

            # Example logic for more generic matching
            if "COBRA" in section:
                qa_pairs.append({
                    "prompt": "What is COBRA and how does it apply?",
                    "completion": content.strip()
                })
            elif "PROCEDURE" in section:
                qa_pairs.append({
                    "prompt": "What are the company's standard procedures?",
                    "completion": content.strip()
                })
            elif "MDB" in section:
                qa_pairs.append({
                    "prompt": "What is the MDB policy?",
                    "completion": content.strip()
                })
            # You can add more rules based on actual content found

        if not qa_pairs:
            logging.warning("No Q&A pairs generated. Check the section matching logic.")
        else:
            logging.info(f"Generated {len(qa_pairs)} Q&A pairs.")
        
        return qa_pairs
    except Exception as e:
        logging.error(f"Error creating Q&A pairs: {e}")
        return []

# Function to save the Q&A pairs to a JSONL file
def save_qa_pairs_to_file(qa_pairs, output_filename):
    try:
        output_path = Path(output_filename)
        with open(output_path, 'w') as file:
            for pair in qa_pairs:
                json_line = json.dumps(pair)
                file.write(json_line + '\n')
        logging.info(f"Q&A pairs successfully saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error saving Q&A pairs to file: {e}")
        return None

# Main function to orchestrate the workflow
def main(pdf_path):
    logging.info("Starting PDF processing...")
    
    # Step 1: Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    if text is None:
        return
    
    # Step 2: Split text into sections
    sections = split_into_sections(text)
    if not sections:
        return
    
    # Step 3: Generate Q&A pairs
    qa_pairs = create_qa_pairs(sections)
    if not qa_pairs:
        return
    
    # Step 4: Save Q&A pairs to JSONL file
    output_filename = 'refined_handbook_qa_pairs.jsonl'
    save_qa_pairs_to_file(qa_pairs, output_filename)

if __name__ == "__main__":
    # Adjust the PDF path to be relative to the script's location
    pdf_path = Path(__file__).parent / 'Handbook.PDF'
    
    # Run the main workflow
    main(pdf_path)
