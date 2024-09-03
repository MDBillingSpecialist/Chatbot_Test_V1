import random
import json

# Seed questions and answers - the original 100 questions with corresponding answers
seed_qa_pairs = [
    {"question": "What is the company's Paid Time Off (PTO) policy?", "answer": "PTO is designed to provide associates with flexible paid time off for needs such as vacation, holidays, personal or family illness, and other activities of the associate's choice."},
    {"question": "How do employees accrue PTO?", "answer": "Full-time associates accrue PTO semi-monthly based on their length of service. PTO is added to the associate's PTO bank when the semi-monthly paycheck is issued."},
    {"question": "How should employees report harassment?", "answer": "Employees should immediately notify their supervisor and the Human Resources director if they experience or witness any form of harassment."},
    {"question": "What is the company's policy on workplace safety?", "answer": "The company is committed to providing a safe workplace and expects all associates to use safe work methods and practices at all times."},
    # Add the rest of your 100 questions and answers here...
]

# Functions to generate variations of questions
def create_variations(qa_pair):
    question = qa_pair["question"]
    answer = qa_pair["answer"]
    variations = [
        {"question": f"Can you explain {question.lower()}", "answer": answer},
        {"question": f"How does the company handle {question.lower()}", "answer": answer},
        {"question": f"What are the rules regarding {question.lower()}", "answer": answer},
        {"question": f"Could you clarify {question.lower()}", "answer": answer},
        {"question": f"In what way does the company approach {question.lower()}", "answer": answer},
    ]
    return variations

def create_scenario_based_questions(qa_pair):
    question = qa_pair["question"]
    answer = qa_pair["answer"]
    base_question = question.lower().replace("the company's", "this company's").replace("company", "this company")
    scenarios = [
        {"question": f"If an employee encounters an issue related to {base_question}, what should they do?", "answer": answer},
        {"question": f"What happens if {base_question} is not followed?", "answer": answer},
        {"question": f"How should {base_question} be handled in an emergency situation?", "answer": answer},
        {"question": f"What are the consequences of not adhering to {base_question}?", "answer": answer},
        {"question": f"How does {base_question} apply in special circumstances?", "answer": answer},
    ]
    return scenarios

def create_rephrased_questions(qa_pair):
    question = qa_pair["question"]
    answer = qa_pair["answer"]
    rephrasings = [
        {"question": f"Can you provide details on {question.lower()}?", "answer": answer},
        {"question": f"What does the company say about {question.lower()}?", "answer": answer},
        {"question": f"What are the guidelines for {question.lower()}?", "answer": answer},
        {"question": f"How is {question.lower()} typically managed?", "answer": answer},
        {"question": f"What should employees know about {question.lower()}?", "answer": answer},
    ]
    return rephrasings

# Main function to generate new questions and answers
def generate_qa_pairs(seed_qa_pairs, num_variations=2, num_scenarios=1, num_rephrasings=1):
    new_qa_pairs = []
    for qa_pair in seed_qa_pairs:
        # Generate variations
        variations = create_variations(qa_pair)
        new_qa_pairs.extend(random.sample(variations, min(len(variations), num_variations)))
        
        # Generate scenario-based questions
        scenarios = create_scenario_based_questions(qa_pair)
        new_qa_pairs.extend(random.sample(scenarios, min(len(scenarios), num_scenarios)))
        
        # Generate rephrased questions
        rephrasings = create_rephrased_questions(qa_pair)
        new_qa_pairs.extend(random.sample(rephrasings, min(len(rephrasings), num_rephrasings)))
    
    return new_qa_pairs

# Generate new questions and answers
generated_qa_pairs = generate_qa_pairs(seed_qa_pairs)

# Output the new questions and answers in the required format
for i, qa_pair in enumerate(generated_qa_pairs, 1):
    print(f"{i}. **Q:** {qa_pair['question']}")
    print(f"   **A:** {qa_pair['answer']}\n")

# Save the new questions and answers to a JSONL file for fine-tuning
with open("generated_qa_pairs.jsonl", "w") as file:
    for qa_pair in generated_qa_pairs:
        file.write(f"{json.dumps(qa_pair)}\n")

print(f"Generated {len(generated_qa_pairs)} Q&A pairs.")
