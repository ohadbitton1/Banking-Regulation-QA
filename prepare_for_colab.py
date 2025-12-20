import json
import os
from sklearn.model_selection import train_test_split

# --- הגדרות ---
INPUT_FILE = "RegulAItion_dataset.json"
OUTPUT_DIR = "FT_datasets"  # שם התיקייה החדשה
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_dataset.json")
TEST_FILE = os.path.join(OUTPUT_DIR, "test_dataset.json")

# פרומפט בפורמט Alpaca (סטנדרט למודלים מסוג Llama)
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
    
    # 1. בדיקת קיום קובץ הקלט
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # 2. טעינת הנתונים
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    formatted_data = []
    
    # 3. המרת הנתונים לפורמט אימון (Prompt + Completion)
    for item in data:
        # יצירת הקלט (השאלה והקונטקסט)
        input_text = PROMPT_TEMPLATE.format(
            context=item['context'],
            question=item['question']
        )
        
        # יצירת הפלט (ה-JSON שהמודל צריך ללמוד להחזיר)
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

    # 4. חלוקה ל-Train ו-Test (מספר קבוע של 50 לטסט)
    train_data, test_data = train_test_split(formatted_data, test_size=50, random_state=42)
    
    print(f"\nStats:")
    print(f"Total examples: {len(data)}")
    print(f"Training set:   {len(train_data)}")
    print(f"Testing set:    {len(test_data)}")

    # 5. יצירת התיקייה ושמירת הקבצים
    os.makedirs(OUTPUT_DIR, exist_ok=True) # יוצר את התיקייה אם היא איננה
    
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)
        
    with open(TEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4)

    print(f"\n✅ Done! Files saved in folder '{OUTPUT_DIR}':")
    print(f"   1. {TRAIN_FILE} (Upload this to Colab)")
    print(f"   2. {TEST_FILE} (Keep safe for evaluation)")

if __name__ == "__main__":
    prepare_data()