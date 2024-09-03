from openai import OpenAI
import os
import json
import logging

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("processing_log.txt"), logging.StreamHandler()])

def generate_subtopics(client, segment, n_subtopics):
    if len(segment.strip()) < 50:
        logging.warning(f"Segment too short to generate subtopics: {segment[:50]}...")
        return "No valid text to generate subtopics."

    prompt = f"""Based on the following text, generate {n_subtopics} subtopics that are related to the content provided.

Text: {segment}

The list must be without numbers, and without any description of the subtopics. The subtopics should be separated by a comma. There must be no other text than the list."""
    response = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content

def generate_questions(client, segment, n_questions):
    if len(segment.strip()) < 50:
        logging.warning(f"Segment too short to generate questions: {segment[:50]}...")
        return "No valid text to generate questions."

    prompt = f"""Based on the following text, generate {n_questions} questions that are relevant to the content provided.

Text: {segment}

The questions should be directly related to the text and should not include any information not found in the text."""
    
    response = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    
    return response.choices[0].message.content

def generate_responses(client, question, segment):
    if len(segment.strip()) < 50:
        logging.warning(f"Segment too short to generate responses: {segment[:50]}...")
        return "No valid text to generate responses."

    prompt = f"""Based on the following text, generate 2 responses to the question provided. The responses should be strictly based on the text.

Text: {segment}

Question: {question}

The responses must not include any information that is not found in the text provided."""
    response = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content

def load_segments_from_json(json_file):
    """Load segments from a JSON file."""
    with open(json_file, 'r') as file:
        segments = json.load(file)
    return segments

def save_qa_to_json(qa_data, output_file):
    """Save Q&A data to a JSON file."""
    with open(output_file, 'w') as file:
        json.dump(qa_data, file, indent=4)
    logging.info(f"Q&A data saved to {output_file}")

def process_segments_from_json(json_file):
    """Process segments for Q&A generation."""
    segments = load_segments_from_json(json_file)
    qa_data = []
    
    for i, segment in enumerate(segments):
        segment_text = segment.get('text', '').strip()
        if not segment_text or len(segment_text) < 50:
            logging.info(f"Skipping segment {i + 1} due to insufficient content.")
            continue
        
        logging.info(f"Processing segment {i + 1}/{len(segments)}...")
        
        subtopics = generate_subtopics(client, segment_text, n_subtopics=5)
        logging.info(f"Subtopics generated: {subtopics}")
        subtopics_list = subtopics.split(",")

        segment_qa = {"segment": segment_text, "subtopics": []}

        for subtopic in subtopics_list:
            subtopic = subtopic.strip()
            if not subtopic:
                continue

            questions = generate_questions(client, segment_text, n_questions=5)
            logging.info(f"Questions generated for subtopic '{subtopic}': {questions}")
            questions_list = questions.split("\n")

            subtopic_data = {"subtopic": subtopic, "questions": []}

            for question in questions_list:
                question = question.strip()
                if not question:
                    continue

                responses = generate_responses(client, question, segment_text)
                logging.info(f"Responses generated for question '{question}': {responses}")
                responses_list = responses.split("\n")

                question_data = {"question": question, "responses": responses_list}
                subtopic_data["questions"].append(question_data)

            segment_qa["subtopics"].append(subtopic_data)

        qa_data.append(segment_qa)

    output_file = os.path.join("output", "qa_data.json")
    save_qa_to_json(qa_data, output_file)
    return output_file
