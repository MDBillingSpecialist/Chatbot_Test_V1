import os
import logging
from dotenv import load_dotenv
import yaml

# Load environment variables from .env file
load_dotenv()

# Ensure the logs directory exists
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

class Config:
    # Load environment variables for API keys
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    MICROSOFT_APP_ID = os.getenv("MicrosoftAppId")
    MICROSOFT_APP_PASSWORD = os.getenv("MicrosoftAppPassword")
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
    DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    
    # Load YAML configuration for models and parameters
    with open("config.yaml", "r") as f:
        yaml_config = yaml.safe_load(f)

    API_BASE_URL = yaml_config['api']['base_url']
    GENERATION_MODEL = yaml_config['models']['generation_model']
    SCORING_MODEL = yaml_config['models']['scoring_model']
    GENERATION_PARAMS = yaml_config['generation_parameters']
    SCORING_PARAMS = yaml_config['scoring_parameters']

# Create a config object
config = Config()

# Setup logging
logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_directory, "processing_log.txt")),
        logging.StreamHandler()
    ]
)
