import PyPDF2
import docx
import os
from .json_handler import save_segments_to_json
import re

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
        print(f"Error extracting text from PDF {pdf_path}: {e}")
    return text

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file."""
    text = []
    try:
        doc = docx.Document(docx_path)
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
    except Exception as e:
        print(f"Error extracting text from DOCX {docx_path}: {e}")
    return "\n".join(text)

def extract_text_from_plain_text(txt_path):
    """Extract text from a plain text file."""
    text = ""
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
    except Exception as e:
        print(f"Error extracting text from plain text file {txt_path}: {e}")
    return text

def extract_text_from_document(file_path):
    """Determine the file type and extract text accordingly."""
    file_extension = file_path.lower()
    if file_extension.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_extension.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_extension.endswith('.txt'):
        return extract_text_from_plain_text(file_path)
    else:
        print(f"Unsupported file type for {file_path}")
        return ""

def segment_text(text):
    """Segment text into larger, more meaningful chunks."""
    # Split by double newlines to get paragraphs
    paragraphs = text.split('\n\n')
    
    # Combine short paragraphs or those related to the same topic
    segments = []
    current_segment = ""
    min_segment_length = 500  # Minimum character length for a segment
    
    for paragraph in paragraphs:
        if len(current_segment) + len(paragraph) < min_segment_length:
            current_segment += paragraph + "\n\n"
        else:
            segments.append(current_segment.strip())
            current_segment = paragraph + "\n\n"
    
    # Add the last segment
    if current_segment:
        segments.append(current_segment.strip())
    
    return segments

def segment_and_save_text(text, output_dir='output'):
    """Segment the text and save the segments to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    segments = segment_text(text)
    output_file = os.path.join(output_dir, 'segments.json')
    save_segments_to_json(segments, output_file)
    print(f"Segments saved to {output_file}")
    return output_file
