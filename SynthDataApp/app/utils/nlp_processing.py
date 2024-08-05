from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def clean_text(text):
    """Clean the text by removing special characters and normalizing whitespace."""
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip().lower()


def advanced_segment_text(text):
    """Advanced segmentation of text into smaller, logical chunks."""
    # Clean text first
    cleaned_text = clean_text(text)
    
    # Segment into sentences
    sentences = re.split(r'(?<=\.)\s', cleaned_text)
    
    # Further segment based on custom logic if needed
    # For example, based on length or specific keywords
    refined_segments = []
    for sentence in sentences:
        if len(sentence.split()) > 10:  # Example condition to split long sentences further
            refined_segments.append(sentence)
        else:
            refined_segments[-1] += ' ' + sentence if refined_segments else sentence

    return refined_segments


def validate_responses(responses, text):
    """Validate responses to ensure they are relevant to the original text."""
    valid_responses = []
    for response in responses:
        if any(keyword.lower() in response.lower() for keyword in text.split()):
            valid_responses.append(response)
    return valid_responses


def process_text_for_openai(text):
    """Prepare text segments for processing with OpenAI."""
    segments = advanced_segment_text(text)
    return segments
