import os
import logging
import json
import PyPDF2
import re

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
    logging.info(f"Extracted {len(text)} characters from document.")
    return text


def chunk_document(text, chunk_size=10000, overlap_size=2000):
    """Chunk the document into manageable pieces with overlap."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    logging.info(f"Document split into {len(chunks)} chunks with overlap.")
    return chunks


def use_llm_for_toc_and_subtopics(client, text_chunk):
    """Use GPT-4o mini to extract the TOC and subtopics."""
    logging.info("Sending request to OpenAI to analyze TOC and subtopics...")
    prompt = (
        "You are analyzing a corporate handbook. This handbook is well-structured with clearly defined sections, subsections, "
        "and page numbers listed in the Table of Contents (TOC). Your task is to extract the TOC and subtopics from the following text, "
        "ensuring that all major sections and their corresponding subsections are captured accurately. Focus on identifying the structure "
        "and organize the output in a clean, readable format that can be used for further processing.\n\n"
        f"Text to analyze:\n\n{text_chunk}"
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
        toc_text = response.choices[0].message.content
        logging.info("TOC and subtopic extraction completed.")
        logging.debug(f"Extracted TOC Text: {toc_text}")
        return toc_text
    except Exception as e:
        logging.error(f"Error during TOC extraction: {e}")
        return None


def extract_toc_entries(toc_text):
    """Extract TOC entries and subtopics from the detected TOC text."""
    logging.debug(f"Full TOC text: {toc_text}")
    
    toc_pattern = r'-\s+(.+?)\s+\d+'  # Matches items like "- Item .................. 1"
    matches = re.findall(toc_pattern, toc_text)
    if not matches:
        logging.warning("No TOC entries matched with the provided pattern.")
    else:
        logging.info(f"TOC entries detected: {matches}")
    return [title.strip() for title in matches]  # Only return section titles


def use_llm_for_segmentation(client, text_chunk, toc_entries):
    """Use GPT-4o mini to segment a text chunk into subtopics based on TOC entries."""
    logging.info(f"Sending request to OpenAI to segment text based on TOC entries...")

    prompt = (
        "You are processing a segment of a corporate handbook. The following text corresponds to sections identified in the "
        "Table of Contents (TOC). Your task is to accurately segment the text based on the TOC entries provided. "
        "For each section, extract only the relevant text. If a section is not present in the text, simply skip it without mentioning. "
        "The output should include only the relevant content organized under the respective section headers from the TOC."
        "\n\nTOC Entries:\n"
        + "\n".join(toc_entries) +
        "\n\nText to analyze:\n\n" + text_chunk
    )
    
    logging.debug(f"Segmenting Chunk: {text_chunk[:500]}...")  # Print the start of the chunk for debugging
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
        segmented_text = response.choices[0].message.content
        logging.info(f"Segmentation completed based on TOC entries.")
        return segmented_text
    except Exception as e:
        logging.error(f"Error during segmentation: {e}")
        return None


def save_segment_to_json(section_title, segment, output_file):
    """Save a single segmented text to a JSON file."""
    data = {section_title: segment}
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        f.write('\n')  # Ensure each segment is on a new line
    logging.info(f"Segment '{section_title}' saved to {output_file}")


def parse_segmented_text(segmented_text, toc_entries):
    """Parse the segmented text into sections based on TOC entries."""
    segments = {}
    current_section = None
    current_content = []

    for line in segmented_text.splitlines():
        line = line.strip()
        if any(line.startswith(f"# {entry}") for entry in toc_entries):  # Assuming the LLM uses Markdown headings
            if current_section:
                segments[current_section] = "\n".join(current_content)
            current_section = line
            current_content = []
        elif current_section:
            current_content.append(line)

    if current_section:  # Don't forget to save the last section
        segments[current_section] = "\n".join(current_content)

    return segments


def segment_based_on_toc_and_subtopics(text, toc_entries, output_file):
    """Segment the document text based on TOC entries and save each segment immediately."""
    chunks = chunk_document(text)
    found_sections = {}

    for chunk in chunks:
        remaining_toc = [entry for entry in toc_entries if entry not in found_sections]
        if not remaining_toc:
            logging.info("All sections have been processed.")
            break

        segmented_text = use_llm_for_segmentation(client, chunk, remaining_toc)
        if segmented_text:
            parsed_segments = parse_segmented_text(segmented_text, remaining_toc)
            for section, content in parsed_segments.items():
                save_segment_to_json(section, content, output_file)
                found_sections[section] = True

    logging.info("Document segmentation process completed.")
def process_document_for_segmentation(file_path):
    """Process the document for chunking and segmentation."""
    text = extract_text_from_pdf(file_path)
    
    toc = None
    chunks = chunk_document(text)
    for chunk in chunks[:3]:  # Check the first few chunks for TOC
        toc = use_llm_for_toc_and_subtopics(client, chunk)
        if toc:
            break

    if toc:
        logging.info("TOC detected.")
        toc_entries = extract_toc_entries(toc)
        if not toc_entries:
            logging.error("TOC detected but no entries were found.")
            return
        segment_based_on_toc_and_subtopics(text, toc_entries, 'output/segments.json')
    else:
        logging.warning("No TOC detected. Falling back on basic segmentation.")
        chunks = chunk_document(text)
        for chunk in chunks:
            save_segment_to_json("Unsegmented Chunk", chunk, 'output/segments.json')

    logging.info("Segmentation process completed.")


# Example Usage
if __name__ == "__main__":
    file_path = 'Handbook.PDF'
    process_document_for_segmentation(file_path)
    print("Document segmentation complete.")