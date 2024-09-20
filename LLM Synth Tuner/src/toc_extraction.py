import re
import fitz
import logging
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(filename='toc_extraction.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_openai_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OpenAI API key not found.")
        return None
    return OpenAI(api_key=api_key)

def extract_text_and_images(pdf_path, start_page=0, num_pages=10):
    content = {"text": "", "images": []}
    try:
        with fitz.open(pdf_path) as pdf_document:
            for page_num in range(start_page, min(start_page + num_pages, pdf_document.page_count)):
                page = pdf_document[page_num]
                content["text"] += page.get_text("text") + "\n"
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_info = {
                        "page": page_num,
                        "type": base_image["ext"],
                        "size": len(base_image["image"])
                    }
                    content["images"].append(image_info)
        return content
    except Exception as e:
        logging.error(f"Error extracting content from PDF: {e}")
        return content

def detect_document_structure(content):
    text = content["text"]
    has_toc = bool(re.search(r'^\s*(?:Table of Contents|Contents|TOC)\s*$', text, re.MULTILINE))
    has_headers = bool(re.findall(r'^\s*(?:\d+\.)*\d+\s+[\w\s]+', text, re.MULTILINE))
    has_images = len(content["images"]) > 0
    
    return {
        "has_toc": has_toc,
        "has_headers": has_headers,
        "has_images": has_images
    }

async def extract_structure_llm(client, content):
    text_sample = content["text"][:4000]  # Limit text to avoid token limits
    image_info = json.dumps(content["images"][:10])  # Limit to first 10 images
    
    prompt = (
        "Analyze this document excerpt and describe its structure. "
        "Identify main sections, subsections, and any notable features like images or graphs. "
        "If a Table of Contents is present, extract it. "
        "If no clear structure is found, suggest logical divisions based on content. "
        "Format the result as a JSON object with the following structure:\n"
        "{\n"
        "  'document_type': 'report/article/manual/etc',\n"
        "  'structure': ['list', 'of', 'main', 'sections'],\n"
        "  'toc': {'section1': {'subsection1': {}, 'subsection2': {}}, 'section2': {}},\n"
        "  'notable_features': ['list', 'of', 'features']\n"
        "}\n\n"
        f"Document text sample:\n\n{text_sample}\n\n"
        f"Image information:\n{image_info}"
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in document analysis and structuring."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4096
        )
        structure_json = json.loads(response.choices[0].message.content)
        logging.info("LLM structure extraction completed successfully.")
        return structure_json
    except Exception as e:
        logging.error(f"Error during LLM structure extraction: {e}")
        return None

async def main_async(pdf_path, output_file):
    client = load_openai_client()
    if not client:
        return

    content = extract_text_and_images(pdf_path)
    document_structure = detect_document_structure(content)
    
    structure_data = await extract_structure_llm(client, content)

    if structure_data:
        structure_data["detected_structure"] = document_structure
        with open(output_file, 'w') as f:
            json.dump(structure_data, f, indent=4)
        logging.info(f"Document structure saved to: {output_file}")
    else:
        logging.error("Structure extraction failed or incomplete.")

if __name__ == "__main__":
    import asyncio
    pdf_path = "path/to/your/document.pdf"
    output_file = "path/to/document_structure.json"
    asyncio.run(main_async(pdf_path, output_file))
