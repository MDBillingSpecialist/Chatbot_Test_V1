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
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
import random
from transformers import pipeline
import numpy as np

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Setup logging configuration
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "False").lower() in ("true", "1", "t") else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_directory, "processing_log.txt")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class ConfigValidator:
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        required_keys = ['models', 'generation_parameters', 'validation', 'augmentation']
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required config key: {key}")
                raise KeyError(f"Missing required config key: {key}")
        logger.debug(f"Configuration validated successfully: {config}")

def load_config(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
        ConfigValidator.validate_config(config)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

# Load configuration
try:
    yaml_config = load_config("config.yaml")
except Exception as e:
    logger.error(f"Failed to load configuration: {str(e)}")
    raise

class QAGenerator:
    def __init__(self):
        self.is_openai_model = "gpt-4" in yaml_config['models']['generation_model'] or "gpt-4" in yaml_config['models']['scoring_model']
        self.client = self._initialize_client()
        self.generation_model = yaml_config['models']['generation_model']
        self.scoring_model = yaml_config['models']['scoring_model']
        self.sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.fluency_checker = pipeline("text-classification", model="textattack/roberta-base-CoLA")

    def _initialize_client(self):
        try:
            if self.is_openai_model:
                return OpenAI(api_key=os.getenv(yaml_config['openai_api']['api_key_env']))
            else:
                return OpenAI(base_url=yaml_config['nvidia_api']['base_url'], api_key=os.getenv(yaml_config['nvidia_api']['api_key_env']))
        except Exception as e:
            logger.error(f"Error initializing API client: {str(e)}")
            raise

    def generate_questions(self, segment_text: str, n_questions: int) -> List[str]:
        question_types = [
            "What is the main point of",
            "How does this policy affect",
            "Can you explain the process for",
            "What are the key requirements for",
            "In what situation would you apply",
            "What are the consequences of not following",
            "How does this compare to",
            "What's an example of",
            "Why is it important to",
            "What steps should be taken if"
        ]
        
        prompt = (f"Given the following text, generate {n_questions} diverse questions:\n"
                  f"Use these question starters: {', '.join(question_types)}\n\n"
                  f"Text: {segment_text}\n\n"
                  f"Instructions:\n"
                  f"1. Ensure questions are directly related to the given text.\n"
                  f"2. Vary the complexity of questions from simple to more advanced.\n"
                  f"3. Include at least one question that requires synthesizing information from different parts of the text.\n"
                  f"4. Avoid repetitive or overly similar questions.\n\n"
                  f"Questions:")
        
        try:
            response = self._get_completion(prompt, self.generation_model)
            questions = response.strip().split('\n')
            logger.info(f"Questions generated: {questions}")
            return questions
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []

    def generate_responses(self, question: str, segment_text: str) -> Dict[str, str]:
        prompt = (
            f"Based on the following text, generate 2 detailed responses to the question: {question}\n\n"
            f"Text: {segment_text}\n\n"
            f"Instructions:\n"
            f"1. Ensure responses are accurate and based on the given text.\n"
            f"2. Make responses detailed and informative.\n"
            f"3. Vary the style and focus between the two responses.\n"
            f"4. Include relevant examples or elaborations where appropriate.\n"
            f"5. Maintain a professional and objective tone.\n\n"
            f"The responses should be in the format:\nRESPONSE A: [Response A text]\nRESPONSE B: [Response B text]"
        )
        
        try:
            response = self._get_completion(prompt, self.generation_model)
            response_parts = response.split("RESPONSE B:")
            if len(response_parts) != 2:
                raise ValueError("Unexpected response format")
            return {
                "response_a": response_parts[0].replace("RESPONSE A:", "").strip(),
                "response_b": response_parts[1].strip()
            }
        except Exception as e:
            logger.error(f"Error generating responses: {str(e)}")
            return {"response_a": "", "response_b": ""}

    def _get_completion(self, prompt: str, model: str) -> str:
        try:
            if "gpt-4" in model:
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

def validate_response_with_sbert(response: str, segment_text: str, sentence_transformer: SentenceTransformer) -> float:
    embeddings = sentence_transformer.encode([segment_text, response])
    cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return cosine_sim.item()

def check_fluency(text: str, fluency_checker) -> float:
    result = fluency_checker(text)
    return result[0]['score'] if result[0]['label'] == 'LABEL_1' else 1 - result[0]['score']

def augment_data(question: str, response: str, qa_generator: QAGenerator) -> List[Dict[str, str]]:
    augmented_data = []
    
    # Original augmentation
    prompt = f"Rephrase the following question and answer pair while maintaining the same meaning:\nQ: {question}\nA: {response}"
    try:
        variation = qa_generator._get_completion(prompt, qa_generator.generation_model)
        var_parts = variation.split('\nA: ')
        if len(var_parts) != 2:
            raise ValueError("Unexpected variation format")
        augmented_data.append({"question": var_parts[0].replace('Q: ', '').strip(), "response": var_parts[1].strip()})
    except Exception as e:
        logger.error(f"Error in data augmentation: {str(e)}")
    
    # Additional augmentation techniques
    augmented_data.extend(synonym_replacement(question, response))
    augmented_data.extend(back_translation(question, response, qa_generator))
    
    return augmented_data

def synonym_replacement(question: str, response: str) -> List[Dict[str, str]]:
    def replace_synonyms(text):
        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        new_words = words.copy()
        for i, (word, pos) in enumerate(pos_tags):
            if pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('VB'):
                synonyms = wordnet.synsets(word)
                if synonyms:
                    synonym = random.choice(synonyms).lemmas()[0].name()
                    new_words[i] = synonym
        return ' '.join(new_words)
    
    return [
        {"question": replace_synonyms(question), "response": response},
        {"question": question, "response": replace_synonyms(response)}
    ]

def back_translation(question: str, response: str, qa_generator: QAGenerator) -> List[Dict[str, str]]:
    def translate(text, target_lang, source_lang='en'):
        prompt = f"Translate the following {source_lang} text to {target_lang}:\n\n{text}"
        return qa_generator._get_completion(prompt, qa_generator.generation_model)
    
    def back_translate(text, intermediate_lang):
        translated = translate(text, intermediate_lang)
        return translate(translated, 'en', intermediate_lang)
    
    languages = ['es', 'fr', 'de']  # Spanish, French, German
    augmented = []
    for lang in languages:
        try:
            augmented.append({
                "question": back_translate(question, lang),
                "response": back_translate(response, lang)
            })
        except Exception as e:
            logger.error(f"Error in back-translation for language {lang}: {str(e)}")
    
    return augmented

def get_incremented_filename(base_name: str, extension: str) -> str:
    version = 1
    while True:
        filename = f"{base_name}_v{version}.{extension}"
        if not os.path.exists(filename):
            return filename
        version += 1

def save_to_jsonl(data: List[Dict[str, Any]], file_path: str):
    try:
        with open(file_path, 'a') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    except Exception as e:
        logger.error(f"Error saving data to JSONL: {str(e)}")

def process_segment(qa_generator: QAGenerator, title: str, segment_text: str, n_questions: int, output_file: str) -> None:
    try:
        # Check if segment_text is a string
        if not isinstance(segment_text, str):
            logger.error(f"Invalid segment_text for '{title}': expected string, got {type(segment_text)}")
            return

        questions = qa_generator.generate_questions(segment_text, n_questions)
        questions = validate_questions(questions)
        results = []

        min_similarity_score = yaml_config.get('validation', {}).get('min_similarity_score', 0.7)
        min_fluency_score = yaml_config.get('validation', {}).get('min_fluency_score', 0.7)

        for question in questions:
            responses = qa_generator.generate_responses(question, segment_text)
            sim_score_a = validate_response_with_sbert(responses["response_a"], segment_text, qa_generator.sentence_transformer)
            sim_score_b = validate_response_with_sbert(responses["response_b"], segment_text, qa_generator.sentence_transformer)
            fluency_score_a = check_fluency(responses["response_a"], qa_generator.fluency_checker)
            fluency_score_b = check_fluency(responses["response_b"], qa_generator.fluency_checker)

            if (sim_score_a >= min_similarity_score and sim_score_b >= min_similarity_score and
                fluency_score_a >= min_fluency_score and fluency_score_b >= min_fluency_score):
                result = {
                    "segment": title,
                    "question": question,
                    "responses": {
                        "response_a": {
                            "response": responses["response_a"],
                            "similarity_score": sim_score_a,
                            "fluency_score": fluency_score_a
                        },
                        "response_b": {
                            "response": responses["response_b"],
                            "similarity_score": sim_score_b,
                            "fluency_score": fluency_score_b
                        }
                    },
                    "metadata": {
                        "source": "synthetic",
                        "generation_model": qa_generator.generation_model,
                        "validation_score": max(sim_score_a, sim_score_b)
                    }
                }
                results.append(result)
                
                # Data augmentation
                augmented_data = augment_data(question, responses["response_a"], qa_generator)
                for aug_item in augmented_data:
                    aug_sim_score = validate_response_with_sbert(aug_item["response"], segment_text, qa_generator.sentence_transformer)
                    aug_fluency_score = check_fluency(aug_item["response"], qa_generator.fluency_checker)
                    if aug_sim_score >= min_similarity_score and aug_fluency_score >= min_fluency_score:
                        aug_result = {
                            "segment": title,
                            "question": aug_item["question"],
                            "responses": {
                                "response_a": {
                                    "response": aug_item["response"],
                                    "similarity_score": aug_sim_score,
                                    "fluency_score": aug_fluency_score
                                }
                            },
                            "metadata": {
                                "source": "augmented",
                                "generation_model": qa_generator.generation_model,
                                "validation_score": aug_sim_score
                            }
                        }
                        results.append(aug_result)
                
                save_to_jsonl(results, output_file)
                print(f"[bold blue]Processed segment:[/bold blue] {title}")
            else:
                logger.warning(f"Responses for question '{question}' did not meet similarity or fluency criteria.")
    except Exception as e:
        logger.error(f"Error processing segment '{title}': {str(e)}")

def validate_questions(questions: List[str]) -> List[str]:
    valid_questions = [q for q in questions if len(q) > 10 and '?' in q]
    if len(valid_questions) < len(questions):
        logger.warning(f"{len(questions) - len(valid_questions)} questions were filtered out due to validation.")
    return valid_questions

def generate_multi_turn_conversation(qa_generator: QAGenerator, initial_question: str, initial_response: str, segment_text: str, num_turns: int = 3) -> List[Dict[str, str]]:
    conversation = [
        {"role": "user", "content": initial_question},
        {"role": "assistant", "content": initial_response}
    ]
    
    for _ in range(num_turns - 1):
        follow_up_prompt = f"Based on the following conversation and context, generate a follow-up question:\n\nContext: {segment_text}\n\nConversation:\n" + "\n".join([f"{turn['role']}: {turn['content']}" for turn in conversation])
        follow_up_question = qa_generator._get_completion(follow_up_prompt, qa_generator.generation_model)
        
        response_prompt = f"Answer the following question based on the given context and conversation history:\n\nContext: {segment_text}\n\nConversation:\n" + "\n".join([f"{turn['role']}: {turn['content']}" for turn in conversation]) + f"\n\nQuestion: {follow_up_question}"
        follow_up_response = qa_generator._get_completion(response_prompt, qa_generator.generation_model)
        
        conversation.extend([
            {"role": "user", "content": follow_up_question},
            {"role": "assistant", "content": follow_up_response}
        ])
    
    return conversation

def main():
    segmented_output_path = r"C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\data\segmented_output\segmented_output.json"
    n_questions = yaml_config.get('generation_parameters', {}).get('n_questions', 5)
    output_dir = r"C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\data\processed"
    base_name = os.path.join(output_dir, "synthetic_data")
    extension = "jsonl"
    output_file = get_incremented_filename(base_name, extension)
    qa_generator = QAGenerator()

    try:
        with open(segmented_output_path, 'r') as f:
            segments = json.load(f)
        
        # Check if segments is a dictionary, if not, try to convert it
        if not isinstance(segments, dict):
            if isinstance(segments, str):
                segments = {"Main Content": segments}
            else:
                raise ValueError(f"Unexpected type for segments: {type(segments)}")
    except Exception as e:
        logger.error(f"Error loading or processing segmented output: {str(e)}")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[green]Generating Q&A pairs...", total=len(segments))
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [
                executor.submit(partial(process_segment, qa_generator, title, segment_text, n_questions, output_file))
                for title, segment_text in segments.items()
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                    progress.update(task, advance=1)
                except Exception as e:
                    logger.error(f"Error processing segment: {str(e)}")

        # Generate multi-turn conversations
        task = progress.add_task("[cyan]Generating multi-turn conversations...", total=len(segments))
        for title, segment_text in segments.items():
            try:
                initial_qa = qa_generator.generate_questions(segment_text, 1)[0]
                initial_response = qa_generator.generate_responses(initial_qa, segment_text)["response_a"]
                conversation = generate_multi_turn_conversation(qa_generator, initial_qa, initial_response, segment_text)
                
                result = {
                    "segment": title,
                    "conversation": conversation,
                    "metadata": {
                        "source": "multi-turn",
                        "generation_model": qa_generator.generation_model
                    }
                }
                save_to_jsonl([result], output_file)
                progress.update(task, advance=1)
            except Exception as e:
                logger.error(f"Error generating multi-turn conversation for segment '{title}': {str(e)}")

    print(f"[bold green]Process completed. Data saved incrementally to {output_file}[/bold green]")

    # Perform final dataset analysis
    analyze_dataset(output_file)

def analyze_dataset(file_path: str):
    try:
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        total_qa_pairs = sum(1 for item in data if "question" in item)
        total_conversations = sum(1 for item in data if "conversation" in item)
        avg_similarity_score = np.mean([item["responses"]["response_a"]["similarity_score"] for item in data if "responses" in item])
        avg_fluency_score = np.mean([item["responses"]["response_a"]["fluency_score"] for item in data if "responses" in item and "fluency_score" in item["responses"]["response_a"]])
        
        print(f"[bold]Dataset Analysis:[/bold]")
        print(f"Total Q&A pairs: {total_qa_pairs}")
        print(f"Total multi-turn conversations: {total_conversations}")
        print(f"Average similarity score: {avg_similarity_score:.2f}")
        print(f"Average fluency score: {avg_fluency_score:.2f}")
        
        # You can add more analysis here, such as distribution of question types, response lengths, etc.
        
    except Exception as e:
        logger.error(f"Error analyzing dataset: {str(e)}")

if __name__ == "__main__":
    main()