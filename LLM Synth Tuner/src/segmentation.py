import re
import pdfplumber
import fitz
import logging
import json
import os
from typing import Dict, List, Tuple
from tqdm import tqdm

logging.basicConfig(filename='segmentation.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

MAX_SEGMENT_TOKENS = 512  # Adjust based on your RAG system's requirements

def load_structure(structure_json_path: str) -> Dict:
    with open(structure_json_path, 'r') as f:
        return json.load(f)

def create_patterns(structure: Dict) -> List[Tuple[str, re.Pattern]]:
    patterns = []
    for section in structure.get('structure', []):
        patterns.append((section, re.compile(rf"^\s*{re.escape(section)}[\s:]*$", re.IGNORECASE | re.MULTILINE)))
    
    toc = structure.get('toc', {})
    for section, subsections in toc.items():
        patterns.append((section, re.compile(rf"^\s*{re.escape(section)}[\s:]*$", re.IGNORECASE | re.MULTILINE)))
        if isinstance(subsections, dict):
            for subsection in subsections:
                patterns.append((subsection, re.compile(rf"^\s*{re.escape(subsection)}[\s:]*$", re.IGNORECASE | re.MULTILINE)))
    
    return patterns

def extract_images(pdf_path: str) -> List[Dict]:
    images = []
    try:
        with fitz.open(pdf_path) as pdf_document:
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    images.append({
                        "page": page_num,
                        "type": base_image["ext"],
                        "size": len(base_image["image"]),
                        "content": base_image["image"]
                    })
    except Exception as e:
        logging.error(f"Error extracting images from PDF: {e}")
    return images

def segment_pdf(pdf_path: str, patterns: List[Tuple[str, re.Pattern]], structure: Dict) -> List[Dict]:
    segments = []
    current_title = "Introduction"
    current_segment = []
    current_tokens = 0
    images = extract_images(pdf_path)
    image_index = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(tqdm(pdf.pages, desc="Processing pages")):
                text = page.extract_text()
                if not text:
                    continue

                lines = text.split('\n')
                for line in lines:
                    matched = False
                    for title, pattern in patterns:
                        if pattern.match(line):
                            if current_segment:
                                segments.append({
                                    "title": current_title,
                                    "content": " ".join(current_segment),
                                    "tokens": current_tokens
                                })
                            current_title = title
                            current_segment = []
                            current_tokens = 0
                            matched = True
                            break

                    if not matched:
                        current_segment.append(line)
                        current_tokens += len(line.split())

                    if current_tokens >= MAX_SEGMENT_TOKENS:
                        segments.append({
                            "title": current_title,
                            "content": " ".join(current_segment),
                            "tokens": current_tokens
                        })
                        current_segment = []
                        current_tokens = 0

                # Check for images on this page
                while image_index < len(images) and images[image_index]["page"] == page_num:
                    img = images[image_index]
                    segments.append({
                        "title": f"Image in {current_title}",
                        "content": f"[Image: {img['type']} format, size: {img['size']} bytes]",
                        "tokens": 20,  # Arbitrary token count for images
                        "image": img["content"]
                    })
                    image_index += 1

    except Exception as e:
        logging.error(f"Error during PDF processing: {e}")

    # Add the last segment if any
    if current_segment:
        segments.append({
            "title": current_title,
            "content": " ".join(current_segment),
            "tokens": current_tokens
        })

    return segments

def post_process_segments(segments: List[Dict]) -> List[Dict]:
    processed_segments = []
    for segment in segments:
        if "image" in segment:
            processed_segments.append(segment)
            continue

        # Remove redundant whitespace
        content = re.sub(r'\s+', ' ', segment['content']).strip()
        
        # Split long segments
        if segment['tokens'] > MAX_SEGMENT_TOKENS:
            words = content.split()
            current_chunk = []
            current_tokens = 0
            for word in words:
                current_chunk.append(word)
                current_tokens += 1
                if current_tokens >= MAX_SEGMENT_TOKENS:
                    processed_segments.append({
                        "title": f"{segment['title']} (Part {len(processed_segments) + 1})",
                        "content": " ".join(current_chunk),
                        "tokens": current_tokens
                    })
                    current_chunk = []
                    current_tokens = 0
            if current_chunk:
                processed_segments.append({
                    "title": f"{segment['title']} (Part {len(processed_segments) + 1})",
                    "content": " ".join(current_chunk),
                    "tokens": current_tokens
                })
        else:
            processed_segments.append({
                "title": segment['title'],
                "content": content,
                "tokens": segment['tokens']
            })
    
    return processed_segments

def save_segments(segments: List[Dict], output_file: str):
    with open(output_file, 'w') as f:
        json.dump(segments, f, indent=2)
    logging.info(f"Segments saved to: {output_file}")

def main(pdf_path: str, structure_json_path: str, output_file: str):
    structure = load_structure(structure_json_path)
    patterns = create_patterns(structure)
    
    raw_segments = segment_pdf(pdf_path, patterns, structure)
    processed_segments = post_process_segments(raw_segments)
    
    save_segments(processed_segments, output_file)

if __name__ == "__main__":
    pdf_path = "path/to/your/document.pdf"
    structure_json_path = "path/to/document_structure.json"
    output_file = "path/to/segmented_output.json"
    main(pdf_path, structure_json_path, output_file)
