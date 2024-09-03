from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import logging
import PyPDF2
import docx
import spacy

# Assuming you have the rest of the functions already defined
# such as extract_text_from_document, segment_text, question_generator, etc.

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the uploaded file
            text = extract_text_from_document(file_path)
            segments = segment_text(text)
            questions = question_generator(client, segments, n_questions=5)
            responses = response_generator(client, questions)
            
            # Render results
            return render_template('results.html', questions=questions, responses=responses)
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
