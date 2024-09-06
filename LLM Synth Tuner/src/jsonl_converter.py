import json
import random

# Paths to the input and output files
input_file = 'C:/Users/theth/OneDrive/Documents/GitHub/Chatbot_Test_V1/LLM Synth Tuner/data/processed/synthetic_data_v4.jsonl'
train_file = 'C:/Users/theth/OneDrive/Documents/GitHub/Chatbot_Test_V1/LLM Synth Tuner/data/processed/train_data.jsonl'
val_file = 'C:/Users/theth/OneDrive/Documents/GitHub/Chatbot_Test_V1/LLM Synth Tuner/data/processed/val_data.jsonl'
test_file = 'C:/Users/theth/OneDrive/Documents/GitHub/Chatbot_Test_V1/LLM Synth Tuner/data/processed/test_data.jsonl'

# Split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

def convert_jsonl_and_split(input_file, train_file, val_file, test_file):
    try:
        data = []
        with open(input_file, 'r') as infile:
            for line in infile:
                data_entry = json.loads(line.strip())
                segment = data_entry.get('segment')
                question = data_entry.get('question')

                if not segment or not question:
                    print(f"Skipping entry with missing 'segment' or 'question': {data_entry}")
                    continue

                # Find the best response for both single-turn and multi-turn
                responses = data_entry.get('responses', {})
                if not responses:
                    print(f"Skipping entry with missing 'responses': {data_entry}")
                    continue

                # If multi-turn conversation
                if data_entry.get('multi_turn', False):
                    messages = [
                        {"role": "system", "content": "You are a highly knowledgeable assistant with expertise in MD Billing policies. Your task is to provide accurate answers strictly based on the information provided in the company's handbook and related documents. If you are unsure or the information is not available, clearly state that the handbook does not cover that particular topic. Avoid making guesses or assumptions."}
                    ]
                    for turn in data_entry['conversation']:
                        messages.append({"role": turn['role'], "content": turn['content']})
                else:
                    best_response = max(responses.values(), key=lambda x: x['similarity_score'])['response']
                    messages = [
                        {"role": "system", "content": "You are a highly knowledgeable assistant with expertise in MD Billing policies. Your task is to provide accurate answers strictly based on the information provided in the company's handbook and related documents. If you are unsure or the information is not available, clearly state that the handbook does not cover that particular topic. Avoid making guesses or assumptions."},
                        {"role": "user", "content": f"Segment: {segment}\nQuestion: {question}"},
                        {"role": "assistant", "content": best_response}
                    ]

                chat_completion = {"messages": messages}
                data.append(chat_completion)

        if not data:
            print("No valid data found to process.")
            return

        # Shuffle and split the data
        random.shuffle(data)
        train_size = int(len(data) * train_ratio)
        val_size = int(len(data) * val_ratio)

        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        def write_jsonl(file_path, dataset):
            with open(file_path, 'w') as outfile:
                for entry in dataset:
                    json.dump(entry, outfile)
                    outfile.write('\n')

        write_jsonl(train_file, train_data)
        write_jsonl(val_file, val_data)
        write_jsonl(test_file, test_data)

        print("Data conversion and splitting completed successfully.")
    
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

# Run the updated conversion function
convert_jsonl_and_split(input_file, train_file, val_file, test_file)


def validate_jsonl(file_path):
    try:
        with open(file_path, 'r') as infile:
            for i, line in enumerate(infile):
                try:
                    # Attempt to parse the JSON
                    entry = json.loads(line.strip())

                    # Check for required structure
                    if 'messages' not in entry:
                        print(f"Error in line {i + 1}: Missing 'messages' key")
                        return False

                    messages = entry['messages']
                    if not isinstance(messages, list) or len(messages) != 3:
                        print(f"Error in line {i + 1}: 'messages' should be a list with 3 items")
                        return False

                    roles = [msg['role'] for msg in messages]
                    if roles != ['system', 'user', 'assistant']:
                        print(f"Error in line {i + 1}: Incorrect message roles")
                        return False

                    # Additional checks
                    for msg in messages:
                        if not isinstance(msg['content'], str):
                            print(f"Error in line {i + 1}: Message content is not a string")
                            return False

                except json.JSONDecodeError as e:
                    print(f"Error in line {i + 1}: JSON decoding error: {e}")
                    return False

        print(f"Validation successful for file: {file_path}")
        return True
    
    except Exception as e:
        print(f"An error occurred during validation: {e}")
        return False

# Run the conversion and split
convert_jsonl_and_split(input_file, train_file, val_file, test_file)

# Validate the resulting files
validate_jsonl(train_file)
validate_jsonl(val_file)
validate_jsonl(test_file)