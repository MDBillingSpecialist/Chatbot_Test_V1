import os
import json
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Load environment variables from .env file
load_dotenv()

# Load configuration from YAML file
with open("config.yaml", "r") as f:
    yaml_config = yaml.safe_load(f)

# Setup logging configuration
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "False").lower() in ("true", "1", "t") else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_directory, "processing_log.txt")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConfigValidator:
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        required_keys = ['models', 'generation_parameters', 'validation']
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required config key: {key}")
                raise KeyError(f"Missing required config key: {key}")
        logger.debug(f"Configuration validated successfully: {config}")

# Validate the configuration
ConfigValidator.validate_config(yaml_config)

class QAGenerator:
    def __init__(self):
        self.is_openai_model = "gpt-4o" in yaml_config['models']['generation_model'] or "gpt-4o" in yaml_config['models']['scoring_model']
        self.client = self._initialize_client()
        self.generation_model = yaml_config['models']['generation_model']
        self.scoring_model = yaml_config['models']['scoring_model']

    def _initialize_client(self):
        if self.is_openai_model:
            return OpenAI(api_key=os.getenv(yaml_config['openai_api']['api_key_env']))
        else:
            return OpenAI(base_url=yaml_config['nvidia_api']['base_url'], api_key=os.getenv(yaml_config['nvidia_api']['api_key_env']))

    def generate_questions(self, segment_text: str, n_questions: int) -> List[str]:
        logger.info(f"Generating {n_questions} questions for the segment.")
        prompt = f"Given the following text, generate {n_questions} questions:\n\n{segment_text}\n\nThe questions should be separated by newline characters."
        try:
            response = self._get_completion(prompt, self.generation_model)
            questions = response.strip().split('\n')
            logger.info(f"Questions generated: {questions}")
            return questions
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []

    def generate_responses(self, question: str, segment_text: str) -> Dict[str, str]:
        logger.info(f"Generating responses for the question: {question}")
        prompt = (
            f"Based on the following text, generate 2 responses to the question: {question}\n\n"
            f"{segment_text}\n\n"
            f"The responses should be in the format:\nRESPONSE A: [Response A text]\nRESPONSE B: [Response B text]"
        )
        try:
            response = self._get_completion(prompt, self.generation_model)
            response_a, response_b = response.split("RESPONSE B:")
            return {
                "response_a": response_a.replace("RESPONSE A:", "").strip(),
                "response_b": response_b.strip()
            }
        except Exception as e:
            logger.error(f"Error generating responses: {str(e)}")
            return {"response_a": "", "response_b": ""}

    def _get_completion(self, prompt: str, model: str) -> str:
        try:
            if "gpt-4o" in model:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=yaml_config['generation_parameters']['temperature'],
                    top_p=yaml_config['generation_parameters']['top_p'],
                    max_tokens=yaml_config['generation_parameters']['max_tokens'],
                )
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in API call: {str(e)}")
            raise

def validate_response_with_sbert(response: str, segment_text: str) -> float:
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode([segment_text, response])
    cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return cosine_sim.item()

def get_incremented_filename(base_name: str, extension: str) -> str:
    version = 1
    while True:
        filename = f"{base_name}_v{version}.{extension}"
        if not os.path.exists(filename):
            return filename
        version += 1

def save_to_jsonl(data: List[Dict[str, Any]], file_path: str):
    with open(file_path, 'a') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def process_segment(qa_generator: QAGenerator, title: str, segment_text: str, n_questions: int, output_file: str) -> None:
    questions = qa_generator.generate_questions(segment_text, n_questions)
    questions = validate_questions(questions)
    results = []

    min_similarity_score = yaml_config.get('validation', {}).get('min_similarity_score', 0.7)

    for question in questions:
        responses = qa_generator.generate_responses(question, segment_text)
        sim_score_a = validate_response_with_sbert(responses["response_a"], segment_text)
        sim_score_b = validate_response_with_sbert(responses["response_b"], segment_text)

        if sim_score_a >= min_similarity_score and sim_score_b >= min_similarity_score:
            result = {
                "segment": title,
                "question": question,
                "responses": {
                    "response_a": {"response": responses["response_a"], "similarity_score": sim_score_a},
                    "response_b": {"response": responses["response_b"], "similarity_score": sim_score_b}
                },
            }
            results.append(result)
            save_to_jsonl([result], output_file)
            print(f"[bold blue]Processed segment:[/bold blue] {title}")
        else:
            logger.warning(f"Responses for question '{question}' did not meet similarity score criteria.")

def validate_questions(questions: List[str]) -> List[str]:
    valid_questions = [q for q in questions if len(q) > 10]
    if len(valid_questions) < len(questions):
        logger.warning(f"Some questions were filtered out due to validation.")
    return valid_questions

def main():
    segmented_output_path = r"C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\data\segmented_output\segmented_output.json"
    n_questions = 5
    output_dir = r"C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\data\processed"
    base_name = os.path.join(output_dir, "synthetic_data")
    extension = "jsonl"
    output_file = get_incremented_filename(base_name, extension)
    qa_generator = QAGenerator()

    with open(segmented_output_path, 'r') as f:
        segments = json.load(f)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[green]Generating Q&A pairs...", total=len(segments))
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(partial(process_segment, qa_generator, title, segment_text, n_questions, output_file))
                for title, segment_text in segments.items()
            ]

            for future in as_completed(futures):
                try:
                    future.result()  # This will raise any exceptions caught in the thread
                    progress.update(task, advance=1)
                except Exception as e:
                    logger.error(f"Error processing segment: {str(e)}")

    print(f"[bold green]Process completed. Data saved incrementally to {output_file}[/bold green]")

if __name__ == "__main__":
    main()
