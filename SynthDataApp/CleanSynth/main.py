import logging
from utils.text_extraction import extract_text_from_pdf
from utils.chunking import chunk_document
from utils.toc_extraction import extract_toc_from_llm_response
from utils.segmentation import segment_based_on_toc_and_subtopics_llm
from utils.json_handling import save_json, load_json
from utils.llm_client import setup_llm_client

def main():
    # Set up logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    file_path = 'Handbook.PDF'  # Ensure this is the correct path

    # Step 1: Extract text from the PDF
    logging.info("Starting text extraction from PDF...")
    text = extract_text_from_pdf(file_path)
    if not text:
        logging.error("No text extracted from PDF, exiting.")
        return
    save_json('output/extracted_text.json', {"text": text})

    # Step 2: Load the extracted text
    extracted_data = load_json('output/extracted_text.json')
    if not extracted_data or 'text' not in extracted_data:
        logging.error("Failed to load extracted text, exiting.")
        return

    # Step 3: Chunk the extracted text
    logging.info("Chunking the extracted text...")
    chunks = chunk_document(extracted_data['text'])
    save_json('output/chunks.json', {"chunks": chunks})

    # Step 4: Extract the Table of Contents (TOC) using the LLM
    logging.info("Extracting Table of Contents (TOC) using LLM...")
    client = setup_llm_client()
    toc_dict = extract_toc_from_llm_response(client, " ".join(chunks[:3]))

    if not toc_dict:
        logging.error("TOC extraction returned an empty response, exiting.")
        return

    # Save the TOC data
    save_json('output/toc.json', toc_dict)

    # Step 5: Segment the document based on the TOC with LLM assistance
    logging.info("Segmenting the document based on TOC with LLM assistance...")
    toc_data = load_json('output/toc.json')
    chunks_data = load_json('output/chunks.json')
    if not toc_data or not chunks_data:
        logging.error("Failed to load TOC or chunks, exiting.")
        return

    segments = segment_based_on_toc_and_subtopics_llm(client, chunks_data['chunks'], toc_data)
    
    if not segments:
        logging.error("No segments were generated, exiting.")
        return

    save_json('output/segments.json', segments)
    logging.info("Document segmentation complete.")

if __name__ == "__main__":
    main()
