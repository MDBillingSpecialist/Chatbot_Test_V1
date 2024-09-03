from rich import print
import os
import json
import logging
import PyPDF2
import docx 
from openai import OpenAI
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import HfApi
import spacy

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Access API keys from environment variables
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
hf_api_key = os.getenv("HUGGINGFACE_HUB_TOKEN")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=nvidia_api_key
)

# 1. Flexible Document Ingestion and Text Extraction

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(docx_path):
    doc = python-docx.Document(docx_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return "\n".join(text)

def extract_text_from_plain_text(txt_path):
    with open(txt_path, 'r') as file:
        text = file.read()
    return text

def extract_text_from_document(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        return extract_text_from_plain_text(file_path)

# 2. Automatic Segmentation
nlp = spacy.load("en_core_web_sm")  # Load a pre-trained NLP model

def segment_text(text):
    doc = nlp(text)
    segments = [sent.text for sent in doc.sents]
    return segments

# 3. Dynamic Question Generation Based on Answers
QUESTION_FROM_ANSWER_PROMPT_TEMPLATE = """\
Given the following excerpt from the employee handbook, generate {n_questions} questions that could be asked based on this text.

The text is: {answer}

The questions should be without any additional text, separated by newline characters.
"""

def generate_questions_from_segment(client, segment, n_questions=5):
    prompt = QUESTION_FROM_ANSWER_PROMPT_TEMPLATE.format(answer=segment, n_questions=n_questions)
    response = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content.split("\n")

# 4. Synthetic Data Generation
RESPONSE_GENERATION_PROMPT_TEMPLATE = """\
Provide two possible responses to the following question:

Question: {question}

Format the response as:

RESPONSE A: Text for response A
RESPONSE B: Text for response B
"""

def generate_responses(client, question):
    prompt = RESPONSE_GENERATION_PROMPT_TEMPLATE.format(question=question)
    response = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content

def question_generator(client, segments, n_questions):
    question_list = []
    for segment in segments:
        questions = generate_questions_from_segment(client, segment, n_questions=n_questions)
        question_list.extend(questions)
    return question_list

def response_generator(client, question_list):
    response_list = []
    for question in question_list:
        response_content = generate_responses(client, question)
        response_list.append({
            "question": question,
            "responses": {
                "response_a": {"response": response_content.split("RESPONSE B:")[0].replace("RESPONSE A:", "").strip()},
                "response_b": {"response": response_content.split("RESPONSE B:")[-1].strip()}
            },
        })
    return response_list

# 5. Scoring Responses and Filtering
def get_scores_from_response(openai_response_template):
    logprobs = openai_response_template.choices[0].logprobs.content
    score_dict = {}
    for score in logprobs:
        score_dict[score.token] = score.logprob
    return score_dict

def get_response_and_scores(client, model, question, response_content):
    messages = [
        {
            "role": "user",
            "content": question
        },
        {
            "role": "assistant",
            "content": response_content
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    scores = get_scores_from_response(response)
    return scores

def process_question_response_pairs(client, model, question_response_list):
    for question_response_pair in question_response_list:
        question = question_response_pair["question"]

        score_a = get_response_and_scores(client, model, question, question_response_pair["responses"]["response_a"]["response"])
        score_b = get_response_and_scores(client, model, question, question_response_pair["responses"]["response_b"]["response"])

        question_response_pair["responses"]["response_a"].update(score_a)
        question_response_pair["responses"]["response_b"].update(score_b)

threshold = 3.0  # Set threshold for filtering low-quality responses

# Save filtered responses with scores to JSONL
def save_filtered_responses(question_response_list, threshold, output_file):
    with open(output_file, 'w') as f:
        for item in question_response_list:
            response_a = item["responses"]["response_a"]
            response_b = item["responses"]["response_b"]
            if response_a["helpfulness"] < threshold and response_b["helpfulness"] < threshold:
                continue
            f.write(json.dumps(response_a))
            f.write('\n')
            f.write(json.dumps(response_b))
            f.write('\n')

# 6. Dataset Creation and Pushing to Hugging Face Hub
def prepare_dataset_for_training(response_list):
    dataset = Dataset.from_list(response_list)
    return DatasetDict({"train": dataset})

def push_dataset_to_hub(dataset_dict, repo_name):
    dataset_dict.push_to_hub(repo_name)

# 7. Main Execution Workflow
def main(file_path, n_questions=5, threshold=3.0, output_file='synthetic_data_with_scores.jsonl', repo_name="your_repo_name"):
    # Step 1: Extract Text
    text = extract_text_from_document(file_path)
    
    # Step 2: Segment Text
    segments = segment_text(text)
    
    # Step 3: Generate Questions
    questions = question_generator(client, segments, n_questions)
    
    # Step 4: Generate Responses
    responses = response_generator(client, questions)
    
    # Step 5: Score and Filter Responses
    process_question_response_pairs(client, "nvidia/nemotron-4-340b-reward", responses)
    
    # Step 6: Save Filtered Data
    save_filtered_responses(responses, threshold, output_file)
    
    # Step 7: Prepare and Push Dataset
    dataset_dict = prepare_dataset_for_training(responses)
    push_dataset_to_hub(dataset_dict, repo_name)

if __name__ == "__main__":
    file_path = "path_to_your_document.pdf"  # Replace with your document path
    main(file_path)
