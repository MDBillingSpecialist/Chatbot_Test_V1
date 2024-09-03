import os
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(filename='fine_tuning.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths to the training and validation files
train_file_path = r'C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\data\output\train_data.jsonl'
val_file_path = r'C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\data\output\val_data.jsonl'

def upload_file(file_path, purpose):
    try:
        logging.info(f"Uploading file {file_path} for purpose: {purpose}")
        file = client.files.create(
            file=open(file_path, "rb"),
            purpose=purpose
        )
        logging.info(f"File uploaded successfully: {file.id}")
        return file.id
    except Exception as e:
        logging.error(f"Failed to upload file {file_path}: {e}")
        raise

def create_fine_tuning_job(train_file_id, val_file_id):
    try:
        logging.info("Creating fine-tuning job.")
        response = client.fine_tuning.jobs.create(
            training_file=train_file_id,
            validation_file=val_file_id,  # Optional, only include if you uploaded a validation file
            model="gpt-4o-mini-2024-07-18",  # Specify the gpt-4o-mini model
            hyperparameters={"n_epochs": 4}  # Number of training epochs
        )
        logging.info(f"Fine-tuning job created successfully: {response.id}")
        return response.id
    except Exception as e:
        logging.error(f"Failed to create fine-tuning job: {e}")
        raise

def monitor_fine_tuning(fine_tune_id, poll_interval=60):
    while True:
        try:
            status = client.fine_tuning.jobs.retrieve(fine_tune_id)
            logging.info(f"Status: {status.status}")
            print(f"Status: {status.status}")

            if status.status in ['succeeded', 'failed']:
                print(f"Fine-tuned model: {getattr(status, 'fine_tuned_model', 'Not yet completed')}")
                logging.info(f"Fine-tuned model: {getattr(status, 'fine_tuned_model', 'Not yet completed')}")

                if getattr(status, 'result_files', None):
                    print(f"Result Files: {status.result_files}")
                    logging.info(f"Result Files: {status.result_files}")
                else:
                    print("No result files yet.")
                    logging.info("No result files yet.")

                # Check for error details if the job failed
                if status.status == 'failed':
                    if hasattr(status, 'events'):
                        for event in status.events:
                            if event.level == 'error':
                                print(f"Error: {event.message}")
                                logging.error(f"Error: {event.message}")

                break
            else:
                if hasattr(status, 'total_steps') and hasattr(status, 'completed_steps'):
                    percentage_complete = (status.completed_steps / status.total_steps) * 100
                    print(f"Progress: {status.completed_steps}/{status.total_steps} steps ({percentage_complete:.2f}% complete)")
                    logging.info(f"Progress: {status.completed_steps}/{status.total_steps} steps ({percentage_complete:.2f}% complete)")
                else:
                    print("Step information is not available yet.")
                    logging.info("Step information is not available yet.")
                    
                print("Fine-tuning is still in progress...")
                logging.info("Fine-tuning is still in progress...")
                time.sleep(poll_interval)
        except Exception as e:
            logging.error(f"Error during monitoring: {e}")
            print(f"Error during monitoring: {e}")
            time.sleep(poll_interval)

def main():
    try:
        train_file_id = upload_file(train_file_path, "fine-tune")
        val_file_id = upload_file(val_file_path, "fine-tune")
        fine_tune_id = create_fine_tuning_job(train_file_id, val_file_id)
        print(f"Fine-tune ID: {fine_tune_id}")
        monitor_fine_tuning(fine_tune_id)
    except Exception as e:
        logging.error(f"Fine-tuning process failed: {e}")
        print(f"Fine-tuning process failed: {e}")

if __name__ == "__main__":
    main()
