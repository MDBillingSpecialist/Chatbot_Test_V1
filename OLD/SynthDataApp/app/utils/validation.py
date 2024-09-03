from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet


nltk.download('wordnet')

def get_synonyms(word):
    """Fetch synonyms for a given word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def validate_responses(responses, text, min_match_threshold=0.1):
    """
    Validate responses by ensuring they are relevant to the original text.
    
    :param responses: List of generated responses.
    :param text: Original text used to generate the responses.
    :param min_match_threshold: Minimum cosine similarity threshold for a response to be considered valid.
    :return: List of valid responses.
    """
    valid_responses = []
    text_keywords = set(text.lower().split())
    
    # Combine original keywords with their synonyms
    expanded_keywords = set()
    for word in text_keywords:
        expanded_keywords.update(get_synonyms(word))
    expanded_keywords.update(text_keywords)
    
    # Use TF-IDF Vectorizer to compute similarity between text and responses
    vectorizer = TfidfVectorizer().fit([text] + responses)
    text_vector = vectorizer.transform([text])
    
    for response in responses:
        response_vector = vectorizer.transform([response])
        similarity = cosine_similarity(text_vector, response_vector).flatten()[0]
        
        # Check if any expanded keywords are present in the response
        if similarity >= min_match_threshold or any(keyword in response.lower() for keyword in expanded_keywords):
            valid_responses.append(response)
    
    return valid_responses
