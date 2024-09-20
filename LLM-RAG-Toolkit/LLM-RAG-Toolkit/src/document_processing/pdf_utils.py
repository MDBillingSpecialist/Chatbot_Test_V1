import os
import fitz  # PyMuPDF
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def extract_toc_from_pdf(pdf_path: str) -> Dict:
    """Extract the table of contents from the PDF."""
    toc_dict = {}
    try:
        with fitz.open(pdf_path) as doc:
            toc = doc.get_toc(simple=False)
            for entry in toc:
                if len(entry) >= 3:
                    level, title, _ = entry[:3]
                    insert_toc_entry(toc_dict, level, title.strip())
        return toc_dict
    except Exception as e:
        logger.error(f"Error extracting TOC from PDF {os.path.basename(pdf_path)}: {e}", exc_info=True)
        return {}

def insert_toc_entry(toc_dict: Dict, level: int, title: str):
    """Insert a TOC entry into the nested dictionary based on its level."""
    current_level = toc_dict
    for _ in range(level - 1):
        if not current_level:
            current_level[title] = {}
            return
        last_key = next(reversed(current_level))
        current_level = current_level[last_key]
    current_level[title] = {}

def extract_images(pdf_path: str) -> List[Dict]:
    """Extract images from the PDF."""
    images = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                image_list = page.get_images(full=True)
                for img in image_list:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    images.append({
                        "page": page_num,
                        "type": base_image.get("ext", ""),
                        "size": len(base_image["image"]),
                        "content": base_image["image"],
                        "width": base_image.get("width", 0),
                        "height": base_image.get("height", 0)
                    })
    except Exception as e:
        logger.error(f"Error extracting images from PDF {os.path.basename(pdf_path)}: {e}", exc_info=True)
    return images

def extract_text_with_headings(pdf_path: str) -> List[Dict]:
    """Extract text from the PDF along with font information."""
    content = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    content.append({
                                        "text": text,
                                        "font_size": span["size"],
                                        "font_flags": span["flags"],
                                        "page_num": page_num
                                    })
        return content
    except Exception as e:
        logger.error(f"Error extracting text from PDF {os.path.basename(pdf_path)}: {e}", exc_info=True)
        return []
