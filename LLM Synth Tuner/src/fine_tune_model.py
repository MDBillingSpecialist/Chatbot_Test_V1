import os
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants for configuration
API_KEY = os.getenv("OPENAI_API_KEY")
TRAIN_FILE_PATH = r'C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\data\output\train_data.jsonl'
VAL_FILE_PATH = r'C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\LLM Synth Tuner\data\output\val_data.jsonl'
MODEL_NAME = "gpt-4o-mini-2024-07-18"
N_EPOCHS = 4
PRICE_PER_TOKEN = 0.0001
POLL_INTERVAL = 60  # in seconds

# Initialize logging
logging.basicConfig(filename='fine_tuning.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the OpenAI client
client = OpenAI(api_key=API_KEY)

def estimate_cost(train_file_path, val_file_path, price_per_token=PRICE_PER_TOKEN):
    try:
        with open(train_file_path, 'r') as train_file, open(val_file_path, 'r') as val_file:
            train_tokens = sum(len(line.split()) for line in train_file)
            val_tokens = sum(len(line.split()) for line in val_file)
        
        total_tokens = train_tokens + val_tokens
        estimated_cost = total_tokens * price_per_token
        return estimated_cost
    except Exception as e:
        logging.error(f"Failed to estimate cost: {e}")
        raise

def upload_file(file_path, purpose):
    try:
        logging.info(f"Uploading file {file_path} for purpose: {purpose}")
        with open(file_path, "rb") as file:
            response = client.files.create(file=file, purpose=purpose)
        logging.info(f"File uploaded successfully: {response.id}")
        return response.id
    except Exception as e:
        logging.error(f"Failed to upload file {file_path}: {e}")
        raise

def create_fine_tuning_job(train_file_id, val_file_id):
    try:
        logging.info("Creating fine-tuning job.")
        response = client.fine_tuning.jobs.create(
            training_file=train_file_id,
            validation_file=val_file_id,
            model=MODEL_NAME,
            hyperparameters={"n_epochs": N_EPOCHS}
        )
        logging.info(f"Fine-tuning job created successfully: {response}")
        return response.id
    except Exception as e:
        logging.error(f"Failed to create fine-tuning job. Full error: {str(e)}")
        raise

def monitor_fine_tuning(fine_tune_id, poll_interval=POLL_INTERVAL):
    while True:
        try:
            status = client.fine_tuning.jobs.retrieve(fine_tune_id)
            logging.info(f"Status: {status.status}")
            print(f"Status: {status.status}")

            if status.status in ['succeeded', 'failed']:
                print(f"Fine-tuned model: {getattr(status, 'fine_tuned_model', 'Not yet completed')}")
                logging.info(f"Fine-tuned model: {getattr(status, 'fine_tuned_model', 'Not yet completed')}")

                if status.result_files:
                    print(f"Result Files: {status.result_files}")
                    logging.info(f"Result Files: {status.result_files}")
                else:
                    print("No result files yet.")
                    logging.info("No result files yet.")

                if status.status == 'failed':
                    print(f"Error: {status.error}")
                    logging.error(f"Error: {status.error}")

                break
            else:
                if status.trained_tokens is not None:
                    print(f"Trained tokens: {status.trained_tokens}")
                    logging.info(f"Trained tokens: {status.trained_tokens}")
                else:
                    print("Token information is not available yet.")
                    logging.info("Token information is not available yet.")
                
                print("Fine-tuning is still in progress...")
                logging.info("Fine-tuning is still in progress...")
                time.sleep(poll_interval)
        except Exception as e:
            logging.error(f"Error during monitoring: {e}")
            print(f"Error during monitoring: {e}")
            time.sleep(poll_interval)

def main():
    try:
        estimated_cost = estimate_cost(TRAIN_FILE_PATH, VAL_FILE_PATH)
        print(f"Estimated cost for fine-tuning: ${estimated_cost:.2f}")
        confirm = input("Do you want to proceed with the fine-tuning? (y/n): ").strip().lower()
        
        if confirm != 'y':
            print("Fine-tuning process aborted.")
            return

        train_file_id = upload_file(TRAIN_FILE_PATH, "fine-tune")
        val_file_id = upload_file(VAL_FILE_PATH, "fine-tune")
        fine_tune_id = create_fine_tuning_job(train_file_id, val_file_id)
        print(f"Fine-tune ID: {fine_tune_id}")
        monitor_fine_tuning(fine_tune_id)
    except Exception as e:
        logging.error(f"Fine-tuning process failed: {e}")
        print(f"Fine-tuning process failed: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")

if __name__ == "__main__":
    main()
