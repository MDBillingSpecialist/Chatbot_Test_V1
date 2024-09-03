from flask import Blueprint, render_template, request, redirect, url_for, current_app, flash
from werkzeug.utils import secure_filename
import os
from app import socketio, db
from app.models import Segment, Subtopic, Question, Response
from app.utils.document_processing import extract_text_from_document, segment_and_save_text
from app.utils.openai_interactions import process_segments_from_json

main = Blueprint('main', __name__)

BATCH_SIZE = 10  # Reduce batch size to test smaller processing chunks

@main.route('/', methods=['GET', 'POST'])
def upload_file():
    print("Upload route accessed")
    if request.method == 'POST':
        print("POST request received")
        file = request.files['file']
        if file and file.filename != '':
            try:
                print(f"File received: {file.filename}")
                filename = secure_filename(file.filename)
                file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                socketio.emit('update', {'progress': 'Starting text extraction'}, namespace='/')
                print("Starting text extraction")

                text = extract_text_from_document(file_path)
                print(f"Text extraction complete, length: {len(text)} characters")

                socketio.emit('update', {'progress': 'Segmenting text'}, namespace='/')
                segments_file = segment_and_save_text(text)  # This saves the segments into a JSON file
                print("Text segmentation completed and saved.")

                print("Processing segments for Q&A")
                qa_file = process_segments_from_json(segments_file)

                print(f"Q&A generation complete. Results saved to {qa_file}")
                return render_template('results.html', qa_file=qa_file)

            except Exception as e:
                print(f"Error during file processing: {e}")
                flash("An error occurred while processing the file. Please try again.", "error")
                return redirect(url_for('main.upload_file'))
        else:
            flash("No file selected or invalid file.", "warning")
            return redirect(url_for('main.upload_file'))

    return render_template('upload.html')
