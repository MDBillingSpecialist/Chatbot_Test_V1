import logging
import json
import os
import time

def segment_based_on_toc_and_subtopics_llm(client, chunks, toc_dict, save_path="segments/combined_segments.json"):
    # Ensure the save directory exists
    save_directory = os.path.dirname(save_path)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    segments = {}
    processed_subtopics = set()  # To track already processed subtopics

    for main_topic, subtopics in toc_dict.items():
        for subtopic, content in subtopics.items():
            if subtopic in processed_subtopics:
                continue  # Skip subtopics that have already been processed

            logging.info(f"Processing subtopic '{subtopic}' under main topic '{main_topic}'.")

            accumulated_content = ""
            segment_found = False

            for chunk_index, chunk in enumerate(chunks):
                # Case-insensitive, space-insensitive matching
                if subtopic.replace(" ", "").lower() in chunk.replace(" ", "").lower():
                    logging.info(f"Found subtopic '{subtopic}' in chunk {chunk_index + 1}")
                    accumulated_content += chunk

                    # Optionally remove the used content from the chunk to avoid duplication
                    chunks[chunk_index] = chunk.replace(accumulated_content, "")

                if accumulated_content:
                    # Construct the prompt with the accumulated content
                    prompt_template = (
                        "Your task is to extract and organize the relevant content from the document based on the subtopic '{subtopic}'. "
                        "Please provide only the content directly related to this subtopic from the provided text. "
                        "Ensure that the extracted segment is coherent and accurately reflects the document's intent.\n\n"
                        "Document Content:\n\n{chunk}"
                    )

                    prompt = prompt_template.format(subtopic=subtopic, chunk=accumulated_content)

                    try:
                        # Send the request to the LLM
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.2,
                            max_tokens=4096
                        )

                        # Extract and clean up the response
                        generated_segment = response.choices[0].message.content.strip()

                        # Check if the response is relevant and add it to the segment
                        if generated_segment and len(generated_segment) > 50:
                            if main_topic not in segments:
                                segments[main_topic] = {}
                            segments[main_topic][subtopic] = generated_segment

                            logging.info(f"Segment for subtopic '{subtopic}' saved successfully.")
                            segment_found = True
                            processed_subtopics.add(subtopic)  # Mark as processed

                            break  # Stop processing chunks for this subtopic

                        # Small delay to handle API rate limits
                        time.sleep(1)

                    except Exception as e:
                        logging.error(f"Error during LLM segmentation for '{subtopic}': {e}")

            if not segment_found:
                logging.warning(f"No content found for subtopic '{subtopic}' in the document.")

    if segments:
        # Save all segments in one JSON file with proper formatting
        with open(save_path, "w", encoding="utf-8") as segment_file:
            json.dump(segments, segment_file, ensure_ascii=False, indent=4)
        logging.info(f"All segments successfully saved to {save_path}.")
    else:
        logging.error("No segments were generated.")
