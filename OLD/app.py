import os
import logging
from flask import Flask, request, jsonify
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure OpenAI configuration
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

def generate_response(prompt):
    """
    Generate a response using the OpenAI GPT model.
    
    Parameters:
        prompt (str): The user prompt to generate a response for.
    
    Returns:
        str: The generated response or an error message.
    """
    try:
        response = openai.ChatCompletion.create(
            engine=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant for the company handbook."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I'm sorry, I encountered an error. Please try again later."

@app.route("/", methods=["GET"])
def root():
    """
    Root endpoint to check the health of the application.
    
    Returns:
        str: A welcome message.
    """
    return "Welcome to the Handbook Chatbot!"

@app.route("/api/messages", methods=["POST"])
def messages():
    """
    Endpoint to handle messages from users and generate responses.
    
    Returns:
        jsonify: A JSON response containing the bot's reply or an error message.
    """
    if request.json and 'text' in request.json:
        user_message = request.json['text']
        bot_response = generate_response(user_message)
        return jsonify({"text": bot_response})
    else:
        return jsonify({"error": "Invalid request"}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3978))
    app.run(host='0.0.0.0', port=port)
