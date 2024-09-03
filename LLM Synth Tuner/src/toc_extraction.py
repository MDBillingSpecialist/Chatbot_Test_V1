import re
import pdfplumber
import logging
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# Setup logging configuration
logging.basicConfig(
    filename='toc_extraction.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def load_openai_client():
    """Load OpenAI client with API key from environment."""
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OpenAI API key not found. Please set it in the .env file.")
        return None
    return OpenAI(api_key=api_key)

def is_probably_toc_page(text):
    """Heuristic to determine if a page is likely part of the TOC."""
    lines = text.split('\n')
    toc_indicators = ['Table of Contents', '.....', 'Chapter', 'Page']
    return any(any(indicator in line for indicator in toc_indicators) for line in lines)

def clean_toc_text(toc_text):
    """Clean up and format the extracted TOC text."""
    lines = toc_text.split('\n')
    cleaned_lines = []

    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
        cleaned_line = re.sub(r'\.{2,}', '', cleaned_line)
        cleaned_lines.append(cleaned_line)

    return '\n'.join(cleaned_lines)

def extract_toc_heuristic(pdf_path):
    """Extract the Table of Contents using a heuristic method."""
    logging.info("Starting heuristic-based TOC extraction process.")
    toc_text = ""
    toc_pages = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            toc_started = False
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    logging.warning(f"No text found on page {page_num}.")
                    continue

                if is_probably_toc_page(text):
                    toc_started = True
                    toc_text += text + "\n"
                    toc_pages.append(page_num)
                    logging.info(f"Identified TOC page: {page_num}")
                elif toc_started and len(toc_text.strip()) > 0:
                    # If TOC has started and current page might still be part of TOC
                    if is_probably_toc_page(text) or len(text.strip()) > 0:
                        toc_text += text + "\n"
                        toc_pages.append(page_num)
                    else:
                        # Stop if TOC is likely finished
                        break

    except Exception as e:
        logging.error(f"Error during heuristic TOC extraction: {e}")
        return None, []

    toc_text = clean_toc_text(toc_text)
    logging.info("Heuristic TOC extraction completed successfully.")
    return toc_text.strip(), toc_pages

def extract_toc_llm(client, text_chunk):
    """Request TOC extraction in JSON format from the LLM."""
    logging.info("Sending request to OpenAI to analyze TOC and subtopics...")

    prompt = (
        "You are analyzing a corporate handbook. This handbook is well-structured with clearly defined sections, subsections, "
        "and page numbers listed in the Table of Contents (TOC). Extract all topics and subtopics, and provide the result in a structured JSON format."
        "\n\nText to analyze:\n\n" + text_chunk
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4096
        )

        raw_toc_string = response.choices[0].message.content
        logging.debug(f"Raw TOC string content: {raw_toc_string}")

        toc_json_start = raw_toc_string.find("{")
        toc_json_end = raw_toc_string.rfind("}") + 1
        toc_json_string = raw_toc_string[toc_json_start:toc_json_end]

        toc_json = json.loads(toc_json_string)
        logging.info("LLM TOC extraction completed successfully.")
        return toc_json

    except Exception as e:
        logging.error(f"Error during LLM TOC extraction: {e}")
        return None

def save_toc(toc_data, output_file):
    """Save the TOC data into a JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(toc_data, f, indent=4)
        logging.info(f"TOC saved to JSON successfully at {output_file}.")
        print(f"TOC data saved to: {output_file}")
        print(json.dumps(toc_data, indent=4))  # Display the TOC in the console for verification
    except Exception as e:
        logging.error(f"Error saving TOC JSON file: {e}")

def main():
    pdf_path = r'C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\data\raw\Handbook.PDF'
    toc_output_file = os.path.join("data", "toc_output.json")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(toc_output_file), exist_ok=True)
    
    # Attempt heuristic-based TOC extraction first
    toc_text, toc_pages = extract_toc_heuristic(pdf_path)
    
    if toc_text:
        save_toc({"toc_text": toc_text, "toc_pages": toc_pages}, toc_output_file)
    else:
        logging.warning("Heuristic TOC extraction failed. Attempting LLM-based extraction.")
        
        # Load OpenAI client and try LLM-based extraction
        client = load_openai_client()
        if client:
            with pdfplumber.open(pdf_path) as pdf:
                first_pages_text = "\n".join([pdf.pages[i].extract_text() for i in range(min(5, len(pdf.pages)))])

            toc_json = extract_toc_llm(client, first_pages_text)
            
            if toc_json:
                save_toc(toc_json, toc_output_file)
            else:
                logging.error("LLM-based TOC extraction failed. No TOC could be extracted.")
        else:
            logging.error("No OpenAI client available for LLM extraction.")
    
    logging.info("TOC extraction script completed.")

if __name__ == "__main__":
    main()
