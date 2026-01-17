import json
import os
import random
from openai import OpenAI

# ================= CONFIGURATION =================
# Set your OpenAI API key
client = OpenAI(api_key="OPENAI_API_KEY")


# Relative paths for VS Code environment
INPUT_CHUNKS = r".\Data\Chunks\Regulatory_Rules_Chunks.json"
OUTPUT_TEST_SET = r".\Data\golden_test_set.json"

# Number of chunks to sample from each unique regulation file
SAMPLES_PER_SOURCE = 3 

def generate_questions():
    """
    Reads chunks and generates a grounded test set using OpenAI GPT-4o.
    """
    if not os.path.exists(INPUT_CHUNKS):
        print(f"❌ Input file not found: {INPUT_CHUNKS}")
        return

    with open(INPUT_CHUNKS, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data.get('chunks', [])
    
    # Group chunks by source to ensure full coverage
    grouped = {}
    for c in chunks:
        src = c['metadata'].get('source', 'Unknown')
        if src not in grouped: grouped[src] = []
        grouped[src].append(c)

    test_set = []
    print(f"🧠 Generating questions from {len(grouped)} source files...")

    for src_path, src_chunks in grouped.items():
        sample = random.sample(src_chunks, min(len(src_chunks), SAMPLES_PER_SOURCE))
        
        for chunk in sample:
            context = chunk['text']
            prompt = f"""
            Based ONLY on the text below, generate 2 questions in English:
            1. POSITIVE: Answer is in text (Verdict: Yes/No).
            2. NEGATIVE: Related topic but NOT in text (Verdict: N.A).
            
            Context: {context}
            
            Return JSON only:
            {{
                "questions": [
                    {{"question": "...", "expected_verdict": "Yes/No", "type": "positive"}},
                    {{"question": "...", "expected_verdict": "N.A", "type": "negative"}}
                ]
            }}
            """
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                res_json = json.loads(response.choices[0].message.content)
                for q in res_json['questions']:
                    q['ground_truth_context'] = context
                    q['source_file'] = os.path.basename(src_path)
                    q['page_label'] = chunk['metadata'].get('page_label', 'N/A')
                    test_set.append(q)
                print(f"✅ Generated for {os.path.basename(src_path)}")
            except Exception as e:
                print(f"❌ Error: {e}")

    with open(OUTPUT_TEST_SET, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, indent=4, ensure_ascii=False)
    print(f"✨ Test set saved to {OUTPUT_TEST_SET}")

if __name__ == "__main__":
    generate_questions()