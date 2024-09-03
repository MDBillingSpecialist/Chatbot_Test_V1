import re
import pdfplumber
import logging
import json
import os

# Setup logging configuration
logging.basicConfig(
    filename='pdf_segmentation.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def create_toc_patterns(toc_structure):
    """Creates a dictionary of compiled regex patterns for all TOC entries."""
    patterns = {}
    for section, subsections in toc_structure.items():
        if isinstance(subsections, dict):
            for subsection in subsections:
                # Improved regex to ensure accurate title detection
                patterns[subsection] = re.compile(f"^{re.escape(subsection)}(?:\\s|$)", re.IGNORECASE)
        else:
            patterns[section] = re.compile(f"^{re.escape(section)}(?:\\s|$)", re.IGNORECASE)
    return patterns

def is_probably_toc_page(text):
    """Heuristic to determine if a page is likely part of the TOC."""
    lines = text.split('\n')
    toc_indicators = ['Table of Contents', '.....', 'Chapter', 'Page']
    return any(any(indicator in line for indicator in toc_indicators) for line in lines)

def validate_segment(title, current_segment):
    """Validate the segment by checking if it logically follows the title."""
    if len(current_segment) < 3:  # Arbitrary number to check for minimal content
        logging.warning(f"Segment for '{title}' is unusually short, may need review.")
    # Post-segmentation validation to ensure no mixed content
    if any(re.search(r"^\d+\.\s", line) for line in current_segment):
        logging.warning(f"Possible incorrect segmentation in '{title}' - Mixed content detected.")
    return current_segment

def segment_pdf(pdf_path, toc_structure):
    """Extract and segment the text from the PDF based on the provided Table of Contents structure."""
    logging.info("Starting PDF segmentation process.")
    text_segments = {}
    current_title = None
    current_segment = []
    toc_skipped = False
    
    # Compile the patterns for each TOC entry
    toc_patterns = create_toc_patterns(toc_structure)
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            logging.info(f"Opened PDF file: {pdf_path}. Total pages: {len(pdf.pages)}")
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    logging.warning(f"No text found on page {page_num}.")
                    continue

                if not toc_skipped:
                    if is_probably_toc_page(text):
                        logging.info(f"Skipping TOC page: {page_num}")
                        continue
                    else:
                        toc_skipped = True

                logging.info(f"Processing page {page_num}.")
                lines = text.split('\n')
                
                for line in lines:
                    # Check if the line matches any TOC entry (either top-level or sub-section)
                    for title, pattern in toc_patterns.items():
                        if pattern.match(line):
                            if current_title and current_segment:
                                text_segments[current_title] = validate_segment(current_title, " ".join(current_segment).strip())
                                current_segment = []
                            current_title = title
                            logging.debug(f"Title detected: '{line.strip()}' - Creating new segment for {current_title}.")
                            break
                    
                    if current_title:
                        current_segment.append(line.strip())
                
                logging.info(f"Finished processing page {page_num}.")
            
            if current_title and current_segment:
                text_segments[current_title] = validate_segment(current_title, " ".join(current_segment).strip())

    except Exception as e:
        logging.error(f"Error during PDF processing: {e}")
        return {}

    logging.info("PDF segmentation completed successfully.")
    return text_segments

def save_segments_to_json(segments, output_file):
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

def main():
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
            "Workers’ Compensation Insurance": "11",
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

    # Use raw string to handle Windows file path
    pdf_path = r'C:\Users\theth\OneDrive\Documents\GitHub\Chatbot_Test_V1\SynthDataApp\CleanSynth\utils\Handbook.PDF'
    output_folder = './segmented_output'
    json_output_file = os.path.join(output_folder, 'segmented_output.json')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logging.info(f"Created output directory: {output_folder}")

    # Extract and segment the text
    segments = segment_pdf(pdf_path, toc_structure)
    
    if segments:
        # Save the segmented text to a JSON file
        save_segments_to_json(segments, json_output_file)
    else:
        logging.warning("No segments to save. The segmentation process may have failed.")

    logging.info("Script execution completed.")

if __name__ == "__main__":
    main()
