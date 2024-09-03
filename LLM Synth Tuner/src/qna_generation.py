import os
import json
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import typer
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv
from openai import OpenAI
from datasets import Dataset, DatasetDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

env_path = r"C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\env\.env"
load_dotenv(dotenv_path=env_path)

class QAGenerator:
    def __init__(self, api_key: str, base_url: str, generation_model: str, scoring_model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.generation_model = generation_model
        self.scoring_model = scoring_model

    def generate_questions(self, segment_text: str, n_questions: int) -> List[str]:
        """Generate questions based on the given segment text."""
        logger.info(f"Generating {n_questions} questions for the segment.")
        prompt = f"Given the following text, generate {n_questions} questions:\n\n{segment_text}\n\nThe questions should be separated by newline characters."
        response = self._get_completion(prompt, self.generation_model)
        questions = response.strip().split('\n')
        logger.info(f"Questions generated: {questions}")
        return questions

    def generate_responses(self, question: str, segment_text: str) -> Dict[str, str]:
        """Generate responses for a given question based on the segment text."""
        logger.info(f"Generating responses for the question: {question}")
        prompt = (
            f"Based on the following text, generate 2 responses to the question: {question}\n\n"
            f"{segment_text}\n\n"
            f"The responses should be in the format:\nRESPONSE A: [Response A text]\nRESPONSE B: [Response B text]"
        )
        response = self._get_completion(prompt, self.generation_model)
        response_a, response_b = response.split("RESPONSE B:")
        return {
            "response_a": response_a.replace("RESPONSE A:", "").strip(),
            "response_b": response_b.strip()
        }

    def _get_completion(self, prompt: str, model: str) -> str:
        """Get completion from the OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in API call: {str(e)}")
            raise

    def get_scores_from_response(self, openai_response_template: Any) -> Dict[str, float]:
        """Extract scores from the OpenAI response."""
        logprobs = openai_response_template.choices[0].logprobs.content
        return {score.token: score.logprob for score in logprobs}

    def get_response_and_scores(self, question: str, response_content: str) -> Dict[str, float]:
        """Get scores for a response to a given question."""
        logger.info(f"Scoring responses for the question: {question}")
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response_content},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.scoring_model,
                messages=messages,
            )
            scores = self.get_scores_from_response(response)
            logger.info(f"Scores obtained: {scores}")
            return scores
        except Exception as e:
            logger.error(f"Error in scoring API call: {str(e)}")
            raise

    def process_question_response_pair(self, pair: Dict[str, Any]) -> Dict[str, Any]:
        """Process and score a question-response pair."""
        question = pair["question"]
        response_a_score = self.get_response_and_scores(question, pair["responses"]["response_a"]["response"])
        response_b_score = self.get_response_and_scores(question, pair["responses"]["response_b"]["response"])

        pair["responses"]["response_a"].update(response_a_score)
        pair["responses"]["response_b"].update(response_b_score)

        return pair

def validate_response(response: str, segment_text: str) -> float:
    """Validate the response against the segment text using cosine similarity."""
    vectorizer = TfidfVectorizer().fit_transform([segment_text, response])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]

def process_segment(qa_generator: QAGenerator, title: str, segment_text: str, n_questions: int, output_file: str) -> None:
    """Process a single segment to generate Q&A pairs and save them immediately."""
    questions = qa_generator.generate_questions(segment_text, n_questions)
    results = []

    for question in questions:
        responses = qa_generator.generate_responses(question, segment_text)
        sim_score_a = validate_response(responses["response_a"], segment_text)
        sim_score_b = validate_response(responses["response_b"], segment_text)

        if sim_score_a >= 0.5 and sim_score_b >= 0.5:
            result = {
                "segment": title,
                "question": question,
                "responses": {
                    "response_a": {"response": responses["response_a"], "similarity_score": sim_score_a},
                    "response_b": {"response": responses["response_b"], "similarity_score": sim_score_b}
                },
            }
            results.append(result)
            # Save the result immediately after processing
            save_to_jsonl([result], output_file)
            print(f"[bold blue]Processed segment:[/bold blue] {title}")

def filter_responses(data: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    """Filter responses based on a helpfulness threshold."""
    filtered_data = []
    for item in data:
        response_a = item["responses"]["response_a"]
        response_b = item["responses"]["response_b"]
        if response_a.get("helpfulness", 0) >= threshold or response_b.get("helpfulness", 0) >= threshold:
            filtered_data.extend([response_a, response_b])
    return filtered_data

def save_to_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data to a JSONL file."""
    with open(file_path, 'a') as f:  # Changed to 'a' to append data incrementally
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    segmented_output_path = r"C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\data\segmented_output\segmented_output.json"
    n_questions = 5
    output_file = r"C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\data\processed\synthetic_data.jsonl"
    
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        logger.error("NVIDIA_API_KEY not found in environment variables.")
        raise typer.Exit(code=1)

    qa_generator = QAGenerator(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
        generation_model="nvidia/nemotron-4-340b-instruct",
        scoring_model="nvidia/nemotron-4-340b-reward"
    )

    with open(segmented_output_path, 'r') as f:
        segments = json.load(f)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[green]Generating Q&A pairs...", total=len(segments))
        with ThreadPoolExecutor() as executor:
            futures = []
            for title, segment_text in segments.items():
                future = executor.submit(partial(process_segment, qa_generator, title, segment_text, n_questions, output_file))
                futures.append(future)

            for future in as_completed(futures):
                progress.update(task, advance=1)

    print(f"[bold green]Process completed. Data saved incrementally to {output_file}[/bold green]")

if __name__ == "__main__":
    main()
