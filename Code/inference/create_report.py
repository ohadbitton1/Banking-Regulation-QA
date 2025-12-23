import json
import csv
import os

# ניסיון לייבא פנדס לעיצוב יפה, אם אין - נשתמש ב-CSV רגיל
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Tip: Install pandas for better formatting (pip install pandas)")

def parse_model_json(json_str):
    """מנסה לפרק את התשובה של המודל ל-JSON אמיתי"""
    try:
        # ניקוי שאריות קוד (כמו ```json) אם יש
        clean_str = json_str.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_str)
    except:
        # במקרה של שגיאה, נחזיר את הטקסט הגולמי
        return {"answer": "Error", "citation": "Error", "explanation": json_str}

def main():
    # --- כאן התיקון לנתיבים ---
    # משיג את הנתיב המלא לתיקייה שבה הסקריפט הזה נמצא כרגע
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # בונה את הנתיב לקובץ ה-JSON באותה תיקייה
    input_file = os.path.join(script_dir, 'evaluation_results.json')
    # בונה את הנתיב לקובץ ה-CSV שייווצר באותה תיקייה
    output_file = os.path.join(script_dir, 'Project_Results_Report.csv')

    print(f"📂 Looking for file at: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Could not find 'evaluation_results.json' in folder:\n{script_dir}")
        print("Please make sure the JSON file is in the same folder as this script.")
        return

    # הכנת השורות לטבלה
    rows = []
    
    print(f"⚙️ Processing {len(data)} items...")

    for item in data:
        # 1. ניתוח תשובת המודל
        model_pred = parse_model_json(item['model_prediction'])
        
        # 2. ניתוח התשובה האמיתית
        true_answer = parse_model_json(item['true_answer'])
        
        # 3. בדיקה האם צדק (KPI 1) - השוואה אדישה לאותיות גדולות/קטנות
        model_ans_str = str(model_pred.get('answer', '')).strip().lower()
        true_ans_str = str(true_answer.get('answer', '')).strip().lower()
        
        is_correct = "✅ Yes" if model_ans_str == true_ans_str else "❌ No"

        rows.append({
            "Question": item['question'],
            "Model Answer": model_pred.get('answer', 'N/A'),
            "True Answer": true_answer.get('answer', 'N/A'),
            "Correct?": is_correct,
            "Model Citation": model_pred.get('citation', 'N/A'),
            "Model Explanation": model_pred.get('explanation', 'N/A')
        })

    # שמירה ל-CSV
    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        # utf-8-sig חובה כדי לראות עברית באקסל כמו שצריך
        df.to_csv(output_file, index=False, encoding='utf-8-sig') 
        print(f"\n📊 Summary:\n{df['Correct?'].value_counts()}")
    else:
        keys = rows[0].keys()
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(rows)

    print(f"\n✅ Report generated successfully!")
    print(f"👉 File saved at: {output_file}")

if __name__ == "__main__":
    main()