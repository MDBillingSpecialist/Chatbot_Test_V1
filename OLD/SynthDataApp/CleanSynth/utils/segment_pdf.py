import re
import pdfplumber
import logging
import json
import os
from typing import Dict, Union, List, Tuple
from fuzzywuzzy import fuzz
import argparse

# Setup logging configuration
logging.basicConfig(
    filename='pdf_segmentation.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def create_toc_patterns(toc_structure: Dict[str, Union[str, Dict[str, str]]]) -> Dict[str, re.Pattern]:
    """Creates a dictionary of compiled regex patterns for all TOC entries."""
    patterns = {}
    for section, subsections in toc_structure.items():
        # Make the pattern more flexible and specific
        pattern = rf"^\s*(?:\d+\s*)?{re.escape(section)}(?:[:\s]|$)"
        patterns[section] = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        if isinstance(subsections, dict):
            for subsection in subsections:
                sub_pattern = rf"^\s*(?:\d+\.\d+\s*)?{re.escape(subsection)}(?:[:\s]|$)"
                patterns[subsection] = re.compile(sub_pattern, re.IGNORECASE | re.MULTILINE)
    return patterns

def fuzzy_match_title(line: str, patterns: Dict[str, re.Pattern], threshold: int = 85) -> Union[str, None]:
    """Perform fuzzy matching on the line against known patterns."""
    for title, pattern in patterns.items():
        if pattern.match(line) or fuzz.ratio(title.lower(), line.lower()) > threshold:
            return title
    return None

def extract_page_numbers(toc_structure: Dict[str, Union[str, Dict[str, str]]]) -> Dict[str, int]:
    """Extract page numbers for each section and subsection."""
    page_numbers = {}
    for section, subsections in toc_structure.items():
        if isinstance(subsections, str):
            page_numbers[section] = int(subsections)
        elif isinstance(subsections, dict):
            for subsection, page in subsections.items():
                page_numbers[subsection] = int(page)
    return page_numbers

def segment_pdf(pdf_path: str, toc_structure: Dict[str, Union[str, Dict[str, str]]]) -> Dict[str, str]:
    """Extract and segment the text from the PDF based on the provided Table of Contents structure."""
    logging.info("Starting PDF segmentation process.")
    text_segments = {}
    current_title = None
    current_segment = []
    page_numbers = extract_page_numbers(toc_structure)
    
    # Compile the patterns for each TOC entry
    toc_patterns = create_toc_patterns(toc_structure)
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logging.info(f"Opened PDF file: {pdf_path}. Total pages: {total_pages}")
            
            for page_num, page in enumerate(pdf.pages, start=1):
                logging.info(f"Processing page {page_num} of {total_pages}")
                text = page.extract_text()
                if not text:
                    logging.warning(f"No text found on page {page_num}.")
                    continue

                lines = text.split('\n')
                
                for line in lines:
                    # Check if the line matches any TOC entry (either top-level or sub-section)
                    title = fuzzy_match_title(line, toc_patterns, threshold=85)
                    if title and page_num >= page_numbers.get(title, 0):
                        if current_title and current_segment:
                            text_segments[current_title] = "\n".join(current_segment).strip()
                            current_segment = []
                        current_title = title
                        logging.debug(f"Title detected: '{line.strip()}' - Creating new segment for {current_title}.")
                    elif current_title:
                        current_segment.append(line.strip())
                
                logging.info(f"Finished processing page {page_num}.")
            
            if current_title and current_segment:
                text_segments[current_title] = "\n".join(current_segment).strip()

    except Exception as e:
        logging.error(f"Error during PDF processing: {e}")
        return {}

    logging.info("PDF segmentation completed successfully.")
    return text_segments

def validate_sections(extracted_sections: Dict[str, str], toc_structure: Dict[str, Union[str, Dict[str, str]]]):
    """Validate if all expected sections are present in the extracted text."""
    missing_sections = []
    for section in toc_structure:
        if section not in extracted_sections:
            missing_sections.append(section)
        elif isinstance(toc_structure[section], dict):
            for subsection in toc_structure[section]:
                if subsection not in extracted_sections:
                    missing_sections.append(f"{section} > {subsection}")
    
    if missing_sections:
        logging.warning(f"Missing sections: {', '.join(missing_sections)}")
        print(f"Warning: The following sections are missing: {', '.join(missing_sections)}")
        print("You may want to review the PDF and the extraction process.")

def save_segments_to_json(segments: Dict[str, str], output_file: str):
    """Save the segmented text into a clean JSON file."""
    if not segments:
        logging.warning("No segments to save. Exiting save function.")
        return

    logging.info(f"Saving segments to JSON file: {output_file}.")
    try:
        with open(output_file, 'w') as f:
            json.dump(segments, f, indent=4)
        logging.info("Segments saved to JSON successfully.")
    except Exception as e:
        logging.error(f"Error saving JSON file: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="PDF Segmentation Tool")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", default="segmented_output.json", help="Output JSON file path")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Define the Table of Contents structure based on the provided JSON outline
    toc_structure = {
        "Introduction": {
            "Corporate Contact Information": "1",
            "Ethics Code": "1",
            "Mission Statement": "2",
            "Values": "2",
            "Goals": "2"
        },
        "Policies and Practices": {
            "Open Door Policy": "3",
            "Employment at Will": "3",
            "Disability Accommodation": "3",
            "Pregnant Workers Fairness Act": "4",
            "Religious Accommodation": "4",
            "Prohibiting Harassment and Discrimination": "5",
            "Policy Against Workplace Harassment": "5",
            "Policy Against Violence": "6",
            "Conflict Resolution": "7",
            "Alternative Dispute Resolution": "8",
            "Employment Status": "8",
            "Outside Employment": "8",
            "Personal Data Changes": "9",
            "Personnel and Medical Records": "9",
            "Employment Resignations": "9",
            "Exit Interview": "9",
            "Post-Employment Reference Policy": "9"
        },
        "Compensation and Benefits": {
            "Benefit Policies": "10",
            "Holiday Observance": "11",
            "Workers' Compensation Insurance": "11",
            "Unemployment Compensation Insurance": "11",
            "New Associates and Introductory Periods": "12",
            "Clinical Associate Evaluations and Assessments": "12",
            "Performance Improvement": "12",
            "Promotions": "13",
            "Pay Raises": "13",
            "Transfer": "13",
            "Workforce Reductions (Layoffs)": "13",
            "Wage and Hours Policies": "14",
            "Revenue Share Compensation": "14",
            "Bonus Compensation": "14",
            "Holiday Differential Compensation": "14",
            "Work Schedules": "14",
            "Time Keeping": "14",
            "Rest and Meal Periods": "15",
            "Off-the Clock Work": "15",
            "Overtime Authorization for Non-exempt Associates": "15",
            "Remote Work Policy": "16",
            "Travel Time Pay": "17",
            "Travel Expenses": "17",
            "Business Expenses": "18",
            "Use of Company Credit Cards": "19",
            "Pay Period": "20",
            "Direct Deposit": "20",
            "Lost or Stolen Paychecks": "20",
            "Paycheck Deductions": "21",
            "Payroll Advances and Loans": "21"
        },
        "Attendance and Leave": {
            "Attendance Policy": "21",
            "Paid Time Off (PTO)": "22",
            "Compensatory Time Off": "26",
            "Emergency Closings": "27",
            "Bereavement Leave": "27",
            "Jury Duty Leave": "28",
            "Court Attendance and Witness Leave": "28",
            "Voting Leave": "28",
            "Leave of Absence": "28"
        },
        "Safety and Compliance": {
            "Employment Requirements": "45",
            "Job Descriptions": "45",
            "Mandatory Training": "45",
            "Mandatory Certification and Licensure Requirements": "45",
            "Safety Policy Statement": "46",
            "Emergency Procedures": "46",
            "Safety Enforcement": "48",
            "Associate Injury and Claims Reporting": "49",
            "Return to Work Policy": "49",
            "Medical Treatment of Associates": "50",
            "Health Insurance Portability & Accountability (HIPAA)": "50",
            "Protection of Property": "51",
            "Intellectual Property": "51",
            "Telephone Use": "52",
            "Mail Use": "52",
            "Company-Provided Mobile Device Policy": "52",
            "Personal Mobile Device Use": "53",
            "Voicemail, Email, and Internet Policy": "53",
            "Social Media Policy": "56",
            "Password Policy and Requirements": "57",
            "Computer Security and Copying of Software": "58",
            "Privacy and Right to Inspect": "58",
            "Vehicles, Equipment, Tools, or Uniforms on Loan": "59",
            "Off-Duty Use of Company Property or Premises": "59",
            "Driving Policy": "59",
            "Personal Appearance": "62",
            "Solicitations and Distribution": "63",
            "Visitors in the Workplace": "64",
            "Company Social Events": "64",
            "Third Party Disclosures": "66"
        },
        "Standards of Conduct": {
            "Disciplinary Process": "66",
            "Standards of Conduct": "67",
            "Progressive Discipline": "68",
            "Immediate Termination": "69",
            "Grievance Procedure": "69",
            "Tobacco and Vape-Free Workplace Policy": "70",
            "Drug and Alcohol Policy": "70",
            "Substance Abuse Policy": "71",
            "Criminal Activity/Arrests": "73",
            "Closing Statement": "73"
        }
    }

    output_folder = './segmented_output'
    os.makedirs(output_folder, exist_ok=True)
    json_output_file = os.path.join(output_folder, args.output)
    
    logging.info(f"Starting PDF segmentation for file: {args.pdf_path}")
    
    # Extract and segment the text
    segments = segment_pdf(args.pdf_path, toc_structure)
    
    if segments:
        # Validate the extracted sections
        validate_sections(segments, toc_structure)
        
        # Save the segmented text to a JSON file
        save_segments_to_json(segments, json_output_file)
        print(f"Segmentation completed. Output saved to: {json_output_file}")
    else:
        logging.warning("No segments to save. The segmentation process may have failed.")
        print("Error: No segments were extracted. Please check the log file for details.")

    logging.info("Script execution completed.")

if __name__ == "__main__":
    main()
