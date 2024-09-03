import logging
import json
from utils.json_handling import save_json, load_json

def extract_toc_from_llm_response(client, text_chunk):
    """Request TOC extraction in JSON format from the LLM and parse the response."""
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

        # Extract and clean up the JSON content from the raw response
        try:
            toc_json_start = raw_toc_string.find("{")
            toc_json_end = raw_toc_string.rfind("}") + 1
            toc_json_string = raw_toc_string[toc_json_start:toc_json_end]

            toc_json = json.loads(toc_json_string)  # Parse the JSON content
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to parse TOC JSON: {e}")
            logging.error(f"Raw TOC string content: {raw_toc_string}")
            return None

        logging.info("TOC successfully parsed and extracted.")
        return toc_json

    except Exception as e:
        logging.error(f"Error during TOC extraction: {e}")
        return None
