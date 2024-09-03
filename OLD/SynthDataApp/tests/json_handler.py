import json

def save_segments_to_json(segments, output_file):
    """Save text segments to a JSON file."""
    data = [{'segment_id': i+1, 'text': segment} for i, segment in enumerate(segments)]
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Segments saved to {output_file}")

def load_segments_from_json(input_file):
    """Load text segments from a JSON file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    return segments

def save_qa_to_json(qa_data, output_file):
    """Save Q&A data to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, indent=4)
    print(f"Q&A data saved to {output_file}")
