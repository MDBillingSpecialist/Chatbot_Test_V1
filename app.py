import os
from flask import Flask, request, jsonify
import openai
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)

# Azure OpenAI configuration
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

def generate_response(prompt):
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
        print(f"Error generating response: {e}")
        return "I'm sorry, I encountered an error. Please try again later."

@app.route("/", methods=["GET"])
def root():
    return "Welcome to the Handbook Chatbot!"

@app.route("/api/messages", methods=["POST"])
def messages():
    if request.json and 'text' in request.json:
        user_message = request.json['text']
        bot_response = generate_response(user_message)
        return jsonify({"text": bot_response})
    else:
        return jsonify({"error": "Invalid request"}), 400

if __name__ == "__main__":
    app.run(port=3978)