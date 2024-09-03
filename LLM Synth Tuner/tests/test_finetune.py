import json
import os
from openai import OpenAI

client = OpenAI()

def validate_file(file_path):
    print(f"Validating file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            line_count += 1
            try:
                data = json.loads(line)
                if 'prompt' not in data or 'completion' not in data:
                    print(f"Error in line {line_count}: Missing 'prompt' or 'completion'")
                    return False
                if not isinstance(data['prompt'], str) or not isinstance(data['completion'], str):
                    print(f"Error in line {line_count}: 'prompt' or 'completion' is not a string")
                    return False
                if not data['completion'].startswith(' '):
                    print(f"Warning in line {line_count}: 'completion' should start with a space")
                if len(data['prompt']) + len(data['completion']) > 4096:
                    print(f"Warning in line {line_count}: Combined prompt and completion may exceed token limit")
                # Check for empty strings
                if not data['prompt'].strip() or not data['completion'].strip():
                    print(f"Error in line {line_count}: 'prompt' or 'completion' is empty")
                    return False
                # Check for consistent prompt structure
                if not data['prompt'].startswith("Segment:") or "\nQuestion:" not in data['prompt']:
                    print(f"Warning in line {line_count}: Prompt structure may be inconsistent")
            except json.JSONDecodeError:
                print(f"Error in line {line_count}: Invalid JSON")
                return False
            except Exception as e:
                print(f"Error in line {line_count}: Unexpected error: {str(e)}")
                return False
    print(f"File validation successful. Total lines: {line_count}")
    return True

def check_api_key():
    try:
        client.models.list()
        print("API key is valid and has correct permissions.")
        return True
    except Exception as e:
        print(f"API key error: {str(e)}")
        return False

def get_file_details(file_id):
    try:
        file = client.files.retrieve(file_id)
        print(f"File details: {file}")
        return file
    except Exception as e:
        print(f"Error retrieving file details: {str(e)}")
        return None

def create_fine_tuning_job(file_id):
    try:
        job = client.fine_tuning.jobs.create(
            training_file=file_id,
            model="gpt-3.5-turbo-1106"
        )
        print(f"Fine-tuning job created: {job}")
        return job.id
    except Exception as e:
        print(f"Error creating fine-tuning job: {str(e)}")
        return None

def troubleshoot():
    if not check_api_key():
        return

    training_file = 'C:\\Users\\theth\\OneDrive\\Documents\\GitHub\\Chatbot_Test_V1\\LLM Synth Tuner\\data\\output\\train_data.jsonl'
    if not validate_file(training_file):
        return

    try:
        with open(training_file, 'rb') as f:
            response = client.files.create(file=f, purpose='fine-tune')
        print(f"File uploaded successfully. File ID: {response.id}")
        
        file_details = get_file_details(response.id)
        if file_details and file_details.status == 'processed':
            job_id = create_fine_tuning_job(response.id)
            if job_id:
                print(f"Fine-tuning job created with ID: {job_id}")
        else:
            print("File processing failed or is still in progress.")
    except Exception as e:
        print(f"Error during troubleshooting: {str(e)}")

if __name__ == "__main__":
    troubleshoot()