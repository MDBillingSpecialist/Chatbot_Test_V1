import logging
import os
import json
from transformers import pipeline
from dotenv import load_dotenv

# Setup logging configuration
logging.basicConfig(
    filename='qna_generation.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def load_llama_model():
    """Load the Hugging Face LLaMA model."""
    load_dotenv()  # Load environment variables from .env file
    model_name = os.getenv("LLAMA_MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
    try:
        generator = pipeline('text-generation', model=model_name)
        return generator
    except Exception as e:
        logging.error(f"Failed to load LLaMA model: {e}")
        return None

def generate_subtopics(generator, text: str, n_subtopics: int = 5) -> str:
    prompt = f"Given the following text, generate {n_subtopics} subtopics:\n\n{text}"
    result = generator(prompt, max_length=1024, num_return_sequences=1)
    subtopics = result[0]['generated_text'].strip()
    logging.info(f"Generated subtopics: {subtopics}")
    return subtopics

def generate_questions(generator, sub_topic: str, n_questions: int = 5) -> str:
    prompt = f"Generate {n_questions} questions related to the following subtopic:\n\n{sub_topic}"
    result = generator(prompt, max_length=1024, num_return_sequences=1)
    questions = result[0]['generated_text'].strip()
    logging.info(f"Generated questions: {questions}")
    return questions

def generate_responses(generator, question: str, text: str) -> str:
    prompt = f"Based on the following text, generate responses to the question: {question}\n\n{text}"
    result = generator(prompt, max_length=1024, num_return_sequences=1)
    responses = result[0]['generated_text'].strip()
    logging.info(f"Generated responses: {responses}")
    return responses

def process_segments_with_llama(segment_file, output_dir):
    # Load the segmented output from JSON
    with open(segment_file, 'r') as f:
        segments = json.load(f)
    
    if segments:
        # Initialize LLaMA model
        generator = load_llama_model()
        if not generator:
            logging.error("Failed to initialize LLaMA model.")
            return
        
        for segment in segments:
            title = segment.get("title", "untitled")
            segment_text = segment.get("text", "")
            logging.info(f"Processing segment: {title}")
            
            # Step 2: Generate subtopics
            subtopics = generate_subtopics(generator, segment_text, n_subtopics=5)
            subtopics_list = subtopics.split("\n")
            
            segment_qa = {"segment_title": title, "segment_text": segment_text, "subtopics": []}
            
            for subtopic in subtopics_list:
                if not subtopic.strip():
                    continue
                
                # Step 3: Generate questions for each subtopic
                questions = generate_questions(generator, subtopic, n_questions=5)
                questions_list = questions.split("\n")
                
                subtopic_data = {"subtopic": subtopic, "questions": []}
                
                for question in questions_list:
                    if not question.strip():
                        continue
                    
                    # Step 4: Generate responses for each question
                    responses = generate_responses(generator, question, segment_text)
                    responses_list = responses.split("\n")
                    
                    question_data = {"question": question, "responses": responses_list}
                    subtopic_data["questions"].append(question_data)
                
                segment_qa["subtopics"].append(subtopic_data)
            
            # Save each segment's Q&A data to a JSON file
            segment_output_file = os.path.join(output_dir, f"{title.replace(' ', '_')}.json")
            os.makedirs(os.path.dirname(segment_output_file), exist_ok=True)
            with open(segment_output_file, 'w') as f:
                json.dump(segment_qa, f, indent=4)
            logging.info(f"Segment Q&A saved to {segment_output_file}")
            print(f"Segment '{title}' processed and saved to: {segment_output_file}")
    else:
        logging.warning("No segments were generated. Please check the PDF or TOC structure.")
        print("Error: No segments were extracted. Please check the log file for details.")

def main():
    segment_file = r"C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\data\processed\segmented_output.json"
    output_dir = r"C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\data\processed"
    
    process_segments_with_llama(segment_file, output_dir)

if __name__ == "__main__":
    main()
