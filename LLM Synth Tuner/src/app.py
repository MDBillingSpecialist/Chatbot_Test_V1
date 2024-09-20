import sys
import os
import streamlit as st
import json
import logging

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fine_tune_model import estimate_cost, upload_file, create_fine_tuning_job, monitor_fine_tuning
from jsonl_converter import convert_jsonl_and_split, validate_jsonl
from qna_generation import QAGenerator, process_segment, generate_multi_turn_conversation, analyze_dataset
from segmentation2 import load_config, segment_pdf_using_toc, segment_pdf_heuristically, save_segments_to_json
from toc_extraction2 import load_openai_client, extract_first_10_pages, extract_toc_llm, save_toc_to_json

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
config_path = 'src/config.yaml'
config = load_config(config_path)

# Streamlit app
st.title("LLM Synth Tuner")

st.sidebar.title("Navigation")
options = ["Fine-Tuning", "Q&A Generation", "PDF Segmentation", "TOC Extraction"]
choice = st.sidebar.selectbox("Choose an option", options)

if choice == "Fine-Tuning":
    st.header("Fine-Tuning")
    
    train_file_path = st.text_input("Training File Path", config['train_file_path'])
    val_file_path = st.text_input("Validation File Path", config['val_file_path'])
    model_name = st.text_input("Model Name", config['model_name'])
    n_epochs = st.number_input("Number of Epochs", min_value=1, max_value=10, value=config['n_epochs'])
    price_per_token = st.number_input("Price per Token", min_value=0.00001, value=config['price_per_token'])
    
    if st.button("Estimate Cost"):
        estimated_cost = estimate_cost(train_file_path, val_file_path, price_per_token)
        st.write(f"Estimated cost for fine-tuning: ${estimated_cost:.2f}")
    
    if st.button("Start Fine-Tuning"):
        try:
            train_file_id = upload_file(train_file_path, "fine-tune")
            val_file_id = upload_file(val_file_path, "fine-tune")
            fine_tune_id = create_fine_tuning_job(train_file_id, val_file_id)
            st.write(f"Fine-tune ID: {fine_tune_id}")
            monitor_fine_tuning(fine_tune_id)
        except Exception as e:
            st.error(f"Fine-tuning process failed: {e}")

elif choice == "Q&A Generation":
    st.header("Q&A Generation")
    
    segmented_output_path = st.text_input("Segmented Output Path", config['segmented_output_path'])
    n_questions = st.number_input("Number of Questions", min_value=1, max_value=20, value=config['n_questions'])
    output_dir = st.text_input("Output Directory", config['output_dir'])
    
    if st.button("Generate Q&A Pairs"):
        qa_generator = QAGenerator()
        with open(segmented_output_path, 'r') as f:
            segments = json.load(f)
        
        for title, segment_text in segments.items():
            process_segment(qa_generator, title, segment_text, n_questions, output_dir)
        
        st.write("Q&A pairs generated successfully.")
    
    if st.button("Analyze Dataset"):
        analyze_dataset(output_dir)
        st.write("Dataset analysis completed.")

elif choice == "PDF Segmentation":
    st.header("PDF Segmentation")
    
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    toc_json_path = st.text_input("TOC JSON Path", config.get('toc_json_path', ''))
    output_dir = st.text_input("Output Directory", config['output_dir'])
    
    if st.button("Segment PDFs"):
        for uploaded_file in uploaded_files:
            pdf_path = os.path.join(output_dir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if toc_json_path and os.path.exists(toc_json_path):
                toc_structure = load_toc(toc_json_path)
                toc_patterns = create_toc_patterns(toc_structure)
                segments = segment_pdf_using_toc(pdf_path, toc_patterns)
            else:
                segments = segment_pdf_heuristically(pdf_path)
            
            output_file = os.path.join(output_dir, f"{os.path.splitext(uploaded_file.name)[0]}_segmented.json")
            if segments:
                save_segments_to_json(segments, output_file)
                st.write(f"Segmentation completed for {uploaded_file.name}. Output saved to: {output_file}")
            else:
                st.error(f"Error: No segments were extracted for {uploaded_file.name}. Please check the log file for details.")

elif choice == "TOC Extraction":
    st.header("TOC Extraction")
    
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    toc_output_dir = st.text_input("TOC Output Directory", config['output_dir'])
    
    if st.button("Extract TOC"):
        client = load_openai_client()
        for uploaded_file in uploaded_files:
            pdf_path = os.path.join(toc_output_dir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            text_chunk = extract_first_10_pages(pdf_path)
            toc_data = extract_toc_llm(client, text_chunk)
            
            toc_output_file = os.path.join(toc_output_dir, f"{os.path.splitext(uploaded_file.name)[0]}_toc.json")
            if toc_data:
                save_toc_to_json(toc_data, toc_output_file)
                st.write(f"TOC saved to: {toc_output_file}")
            else:
                st.error(f"TOC extraction failed or incomplete for {uploaded_file.name}.")