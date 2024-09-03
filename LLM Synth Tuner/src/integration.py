import logging
from src.segmentation import segment_pdf, save_segments_to_json
from src.qna_generation import generate_subtopics, generate_questions, generate_responses
from openai import OpenAI
import os
import json

def process_segments_with_llama(pdf_path, toc_structure, output_file):
    # Step 1: Segment the PDF
    segments = segment_pdf(pdf_path, toc_structure)
    
    if segments:
        # Step 2: Initialize the Q&A data structure
        qa_data = []
        
        # Initialize LLaMA model API client
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY")
        )
        
        for title, segment_text in segments.items():
            logging.info(f"Processing segment: {title}")
            
            # Step 3: Generate subtopics
            subtopics = generate_subtopics(client, segment_text, n_subtopics=5)
            subtopics_list = subtopics.split(", ")
            
            segment_qa = {"segment": segment_text, "subtopics": []}
            
            for subtopic in subtopics_list:
                if not subtopic.strip():
                    continue
                
                # Step 4: Generate questions for each subtopic
                questions = generate_questions(client, subtopic, n_questions=5)
                questions_list = questions.split("\n")
                
                subtopic_data = {"subtopic": subtopic, "questions": []}
                
                for question in questions_list:
                    if not question.strip():
                        continue
                    
                    # Step 5: Generate responses for each question
                    responses = generate_responses(client, question, segment_text)
                    responses_list = responses.split("\n")
                    
                    question_data = {"question": question, "responses": responses_list}
                    subtopic_data["questions"].append(question_data)
                
                segment_qa["subtopics"].append(subtopic_data)
            
            # Add the Q&A data for this segment to the overall dataset
            qa_data.append(segment_qa)
        
        # Step 6: Save the Q&A data to a JSON file
        save_segments_to_json(qa_data, output_file)
        logging.info(f"Q&A data saved to {output_file}")
        print(f"Q&A generation completed. Output saved to: {output_file}")
    else:
        logging.warning("No segments were generated. Please check the PDF or TOC structure.")
        print("Error: No segments were extracted. Please check the log file for details.")

def main():
    # Define the Table of Contents structure
    toc_structure = {
        # Your TOC structure here...
    }
    
    pdf_path = "path/to/your/document.pdf"
    output_file = "path/to/your/output.json"
    
    process_segments_with_llama(pdf_path, toc_structure, output_file)

if __name__ == "__main__":
    main()
