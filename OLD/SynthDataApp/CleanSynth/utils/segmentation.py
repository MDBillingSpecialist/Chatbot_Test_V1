import logging
import json
import os
import time
from typing import List, Dict

def segment_based_on_toc_and_subtopics_llm(client, chunks: List[str], toc_dict: Dict[str, Dict[str, str]], save_path: str = "segments/combined_segments.json"):
    # Ensure the save directory exists
    save_directory = os.path.dirname(save_path)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    segments = {}
    processed_chunks = set()  # To track already processed chunks

    for main_topic, subtopics in toc_dict.items():
        logging.info(f"Processing main topic: {main_topic}")
        segments[main_topic] = {"content": "", "subtopics": {}, "metadata": {"processed_subtopics": []}}

        # First, process the main topic
        main_topic_chunks = []
        for chunk_index, chunk in enumerate(chunks):
            if chunk_index in processed_chunks:
                continue
            if main_topic.lower() in chunk.lower():
                main_topic_chunks.append((chunk_index, chunk))
                processed_chunks.add(chunk_index)

        if main_topic_chunks:
            main_topic_content = "\n".join(chunk for _, chunk in main_topic_chunks)
            segments[main_topic]["content"] = extract_relevant_content(client, main_topic, main_topic_content)

        # Then process each subtopic
        for subtopic in subtopics:
            logging.info(f"Processing subtopic '{subtopic}' under main topic '{main_topic}'.")

            relevant_chunks = []
            for chunk_index, chunk in enumerate(chunks):
                if chunk_index in processed_chunks:
                    continue
                if subtopic.lower() in chunk.lower():
                    relevant_chunks.append((chunk_index, chunk))
                    processed_chunks.add(chunk_index)

            if relevant_chunks:
                accumulated_content = "\n".join(chunk for _, chunk in relevant_chunks)
                segments[main_topic]["subtopics"][subtopic] = extract_relevant_content(client, subtopic, accumulated_content)
                logging.info(f"Segment for subtopic '{subtopic}' saved successfully.")
            else:
                logging.warning(f"No content found for subtopic '{subtopic}' in the handbook.")
                segments[main_topic]["subtopics"][subtopic] = "No relevant content found."

        segments[main_topic]["metadata"]["processed_subtopics"] = list(segments[main_topic]["subtopics"].keys())

    if any(segments[main_topic]["subtopics"] for main_topic in segments):
        with open(save_path, "w", encoding="utf-8") as segment_file:
            json.dump(segments, segment_file, ensure_ascii=False, indent=4)
        logging.info(f"All segments successfully saved to {save_path}.")
    else:
        logging.error("No segments were generated. Please check logs for details.")

    return segments

def extract_relevant_content(client, topic, content):
    prompt_template = (
        "Your task is to extract and organize the relevant content from the handbook based on the topic '{topic}'. "
        "Please provide only the content directly related to this topic from the provided text. "
        "Do not generate new content or summarize. Extract and organize the existing content as is.\n\n"
        "Handbook Content:\n\n{content}"
    )

    prompt = prompt_template.format(topic=topic, content=content)

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant tasked with accurately segmenting handbook content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4096
        )

        extracted_content = response.choices[0].message.content.strip()

        if extracted_content and len(extracted_content) > 50:
            return extracted_content
        else:
            logging.warning(f"Extracted content for '{topic}' is too short or empty.")
            return content  # Return original content if extraction fails

    except Exception as e:
        logging.error(f"Error during LLM content extraction for '{topic}': {e}")
        return content  # Return original content if extraction fails

    finally:
        time.sleep(1)  # Rate limiting