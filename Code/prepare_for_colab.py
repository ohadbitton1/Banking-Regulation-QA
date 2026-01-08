import json
import os
from sklearn.model_selection import train_test_split

# --- Settings ---
INPUT_FILE = "RegulAItion_dataset.json"
OUTPUT_DIR = "../Data/DT_datasets/FT_datasets" # Output Directory
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_dataset.json")
TEST_FILE = os.path.join(OUTPUT_DIR, "test_dataset.json")

# Alpaca prompt
PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a banking regulation compliance assistant. 
Based strictly on the provided context, answer the user's question in JSON format.
Required fields: "answer" (Yes/No), "citation" (verbatim quote), "explanation" (concise justification).

### Input:
Context: {context}
Question: {question}

### Response:
"""

def prepare_data():
    print(f"--- Preparing Data for Llama-3 Training ---")
    
    # 1. Check input file existence
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # 2. Load data
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    formatted_data = []
    
    # 3. Data conversion to a training format (Prompt + Completion)
    for item in data:
        # Creating the input (the question and the context)
        input_text = PROMPT_TEMPLATE.format(
            context=item['context'],
            question=item['question']
        )
        
        # Creating the output (the JSON that the model should learn to return)
        output_json = {
            "answer": item['answer'],
            "citation": item['citation'],
            "explanation": item['explanation']
        }
        output_text = json.dumps(output_json, ensure_ascii=False)
        
        formatted_data.append({
            "instruction": input_text,
            "output": output_text
        })

    # 4. Splitting into Train and Test sets (fixed test size of 50)
    train_data, test_data = train_test_split(formatted_data, test_size=50, random_state=42)
    
    print(f"\nStats:")
    print(f"Total examples: {len(data)}")
    print(f"Training set:   {len(train_data)}")
    print(f"Testing set:    {len(test_data)}")

    # 5. Creating the directory and saving the files
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)
        
    with open(TEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4)

    print(f"\n✅ Done! Files saved in folder '{OUTPUT_DIR}':")
    print(f"   1. {TRAIN_FILE} (Upload this to Colab)")
    print(f"   2. {TEST_FILE} (Keep safe for evaluation)")

if __name__ == "__main__":
    prepare_data()