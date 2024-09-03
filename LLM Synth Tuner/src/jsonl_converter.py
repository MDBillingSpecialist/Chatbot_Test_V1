import json
import random

# Paths to the input and output files
input_file = 'C:/Users/theth/OneDrive/Documents/GitHub/Chatbot_Test_V1/LLM Synth Tuner/data/processed/synthetic_data.jsonl'
train_file = 'C:/Users/theth/OneDrive/Documents/GitHub/Chatbot_Test_V1/LLM Synth Tuner/data/processed/train_data.jsonl'
val_file = 'C:/Users/theth/OneDrive/Documents/GitHub/Chatbot_Test_V1/LLM Synth Tuner/data/processed/val_data.jsonl'
test_file = 'C:/Users/theth/OneDrive/Documents/GitHub/Chatbot_Test_V1/LLM Synth Tuner/data/processed/test_data.jsonl'

# Split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

def convert_jsonl_and_split(input_file, train_file, val_file, test_file):
    # Read and process data
    data = []
    with open(input_file, 'r') as infile:
        for line in infile:
            data_entry = json.loads(line.strip())
            segment = data_entry.get('segment')
            question = data_entry.get('question')

            # Find the best response (you can adjust the logic here)
            best_response = max(data_entry['responses'].values(), key=lambda x: x['similarity_score'])['response']

            # Create the prompt and completion
            prompt = f"Segment: {segment}\nQuestion: {question}"
            completion = f" {best_response}"

            # Format into the desired structure
            output_data = {
                "prompt": prompt,
                "completion": completion
            }

            data.append(output_data)
    
    # Shuffle the data to ensure random distribution
    random.shuffle(data)
    
    # Split the data
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # Write the split datasets to their respective files
    def write_jsonl(file_path, dataset):
        with open(file_path, 'w') as outfile:
            for entry in dataset:
                json.dump(entry, outfile)
                outfile.write('\n')

    write_jsonl(train_file, train_data)
    write_jsonl(val_file, val_data)
    write_jsonl(test_file, test_data)

def validate_jsonl(file_path):
    with open(file_path, 'r') as infile:
        for i, line in enumerate(infile):
            try:
                # Attempt to parse the JSON
                entry = json.loads(line.strip())

                # Check for required keys
                if 'prompt' not in entry or 'completion' not in entry:
                    print(f"Error in line {i + 1}: Missing 'prompt' or 'completion'")
                    return False

                # Additional checks
                if not isinstance(entry['prompt'], str) or not isinstance(entry['completion'], str):
                    print(f"Error in line {i + 1}: 'prompt' or 'completion' is not a string")
                    return False

            except json.JSONDecodeError as e:
                print(f"Error in line {i + 1}: JSON decoding error: {e}")
                return False

    print(f"Validation successful for file: {file_path}")
    return True

# Run the conversion and split
convert_jsonl_and_split(input_file, train_file, val_file, test_file)

# Validate the resulting files
validate_jsonl(train_file)
validate_jsonl(val_file)
validate_jsonl(test_file)
