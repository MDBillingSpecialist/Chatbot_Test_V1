from rich import print
import os
import json
import logging
import PyPDF2
from openai import OpenAI
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import HfApi

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

# Replace "Machine Learning" with a section of the handbook
topic = "Employee Handbook Policies"
n_subtopics = 5  # Number of subtopics you'd like to generate
n_questions = 5  # Number of questions per subtopic

# 1. PDF Text Extraction

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# 2. Subtopics Generation

TOPIC_GENERATION_PROMPT_TEMPLATE = """\
Given the topic of {topic}, generate a list of {n_subtopics} subtopics or sections that could be included in the handbook.

The topic is: {topic}

The list must be without numbers, and without any description of the subtopics. The subtopics should be separated by a comma. There must be no other text than the list.
"""

def generate_subtopics(client, topic, n_subtopics):
    prompt = TOPIC_GENERATION_PROMPT_TEMPLATE.format(topic=topic, n_subtopics=n_subtopics)
    response = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[
            {"role": "user",
             "content": prompt}
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    return response

responses = generate_subtopics(client, topic=topic, n_subtopics=n_subtopics)
print(responses.choices[0].message.content)

# 3. Questions Generation

QUESTION_PROMPT_TEMPLATE = """\
Given the subtopic, generate {n_questions} questions that could be asked about that subtopic in the employee handbook. Your response should be in a list format.

The subtopic is: {sub_topic}

The list must be without numbers. The questions should be separated by a newline character. There must be no other text than the list.
"""

subtopic_list = responses.choices[0].message.content.split(",")

def generate_questions(client, sub_topic, n_questions):
    prompt = QUESTION_PROMPT_TEMPLATE.format(sub_topic=sub_topic, n_questions=n_questions)
    response = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[
            {"role": "user",
             "content": prompt}
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def question_generator(client, subtopic_list, n_questions):
    tasks = [generate_questions(client, subtopic.strip(), n_questions) for subtopic in subtopic_list]
    question_list = tasks
    return question_list

question_list = question_generator(client, subtopic_list, n_questions)
print(question_list)

question_list_formatted = []
for question_set in question_list:
    question_list_formatted.extend([question.strip() for question in question_set.split("\n") if question])
len(question_list_formatted)

# 4. Responses Generation
RESPONSE_PROMPT_TEMPLATE = """\
Given a question from the employee handbook, generate 2 responses that could be provided. Your response should be in a list format.

The question is: {question}

The list must be in the format:

RESPONSE A: Response A text here
RESPONSE B: Response B text here
"""

def generate_responses(client, question):
    prompt = RESPONSE_PROMPT_TEMPLATE.format(question=question)
    response = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[
            {"role": "user",
             "content": prompt}
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def response_generator(client, question_list):
    tasks = [generate_responses(client, question) for question in question_list]
    response_list = tasks
    return response_list

question_response_list = response_generator(client, question_list_formatted)
question_response_pair_list = []
for question, response_set in zip(question_list_formatted, question_response_list):
    question_response_pair_list.append(
        {
            "question": question,
            "responses": {
                "response_a": {"response": response_set.split("RESPONSE B:")[0].replace("RESPONSE A:", "").strip()},
                "response_b": {"response": response_set.split("RESPONSE B:")[-1].split("\n\n")[0].strip()}
            },
        }
    )

# Save the generated data to a JSONL file
with open('synthetic_handbook_data.jsonl', 'w') as f:
    for item in question_response_pair_list:
        f.write(json.dumps(item))
        f.write('\n')

# 5. Scoring Responses and Pushing to Hugging Face

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

question_response_score_list = question_response_pair_list.copy()

def process_question_response_pairs(client, model, question_response_score_list):
    tasks = []
    for question_response_pair in question_response_score_list:
        question = question_response_pair["question"]

        task_a = get_response_and_scores(client, model, question, question_response_pair["responses"]["response_a"]["response"])
        task_b = get_response_and_scores(client, model, question, question_response_pair["responses"]["response_b"]["response"])

        tasks.append((task_a, question_response_pair, "response_a"))
        tasks.append((task_b, question_response_pair, "response_b"))

    results = [task[0] for task in tasks]

    for i, (result, task_info) in enumerate(zip(results, tasks)):
        _, question_response_pair, response_key = task_info
        question_response_pair["responses"][response_key].update(result)

process_question_response_pairs(client, "nvidia/nemotron-4-340b-reward", question_response_score_list)

threshold = 3.0

# Save filtered responses with scores to JSONL
with open(f'synthetic_handbook_data_with_scores_filtered-{threshold}.jsonl', 'w') as f:
    for item in question_response_score_list:
        question = item["question"]
        response_a = item["responses"]["response_a"]
        response_b = item["responses"]["response_b"]
        response_a["question"] = question
        response_b["question"] = question
        if response_a["helpfulness"] < threshold and response_b["helpfulness"] < threshold:
            continue
        f.write(json.dumps(response_a))
        f.write('\n')
        f.write(json.dumps(response_b))
        f.write('\n')

# Load data into a Hugging Face dataset and push to the hub
with open(f'synthetic_handbook_data_with_scores_filtered-{threshold}.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

dataset = Dataset.from_list(data)
dataset_dict = DatasetDict({"train": dataset})

# Push dataset to Hugging Face Hub
dataset_dict.push_to_hub("notthird/ChatBot")
