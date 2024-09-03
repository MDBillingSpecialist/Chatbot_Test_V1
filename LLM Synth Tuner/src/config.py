from dotenv import load_dotenv
import logging
import os

# Load environment variables from .env file
load_dotenv()

class Config:
    # Azure OpenAI settings
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    # Microsoft Bot Framework settings
    MICROSOFT_APP_ID = os.getenv("MicrosoftAppId")
    MICROSOFT_APP_PASSWORD = os.getenv("MicrosoftAppPassword")

    # NVIDIA API settings
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

    # Hugging Face settings
    HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

    # Debug mode
    DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

# Create a config object
config = Config()

logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/processing_log.txt"),
        logging.StreamHandler()
    ]
)