import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

INPUT_FILE = "bank_employee_dataset.json"

# שימוש במודל המקומי
llm_judge = ChatOllama(model="llama3.2", temperature=0)

# פרומפט פשוט יותר: YES / NO (יותר קל למודלים קטנים להבין)
judge_template = """
You are a helpful assistant validating data.

1. Question: "{question}"
2. Answer: "{answer}"
3. Citation: "{citation}"

Does the citation support the answer?
Reply with ONE WORD only: "YES" or "NO".
"""

judge_prompt = PromptTemplate(template=judge_template, input_variables=["question", "answer", "citation"])
chain = judge_prompt | llm_judge | StrOutputParser()

def debug_evaluation():
    print(f"--- Debugging Evaluation (Llama 3.2) ---")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Input file not found.")
        return

    print(f"Testing first 5 items to see what happens...\n")

    # נבדוק רק 5 פריטים כדי לא לבזבז זמן
    for i, item in enumerate(data[:5]):
        print(f"--- Item {i+1} ---")
        print(f"Question: {item['question']}")
        
        try:
            # קבלת תשובה גולמית מהמודל
            raw_response = chain.invoke({
                "question": item['question'],
                "answer": item['answer'],
                "citation": item['citation']
            })
            
            # ניקוי בסיסי
            clean_response = raw_response.strip().upper()
            
            print(f"Model Raw Output: '{raw_response}'") # נראה מה הוא *באמת* אמר
            
            # בדיקה מקלה - מחפשים אם המילה YES מופיעה
            if "YES" in clean_response:
                print("✅ Verdict: PASS")
            else:
                print("❌ Verdict: FAIL")
                
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 30)

if __name__ == "__main__":
    debug_evaluation()