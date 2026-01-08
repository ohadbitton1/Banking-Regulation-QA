import json
import random
import re
import os
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prepare_data import load_and_split_pdfs

# 1. Paste your API key here (hidden for security)
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
# 2. Set the number of chunks to sample (randomly)
SAMPLE_SIZE = 5 # Start small example to see it works well
# If it was good, make sure to adjust quantity of chunks and questions accordingly.
# ==========================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Updated prompt: request a single unified "source" field from the model
prompt_template = """
You are generating **NOISY but legally correct** training data for banking regulation compliance.

Based **STRICTLY** on the provided context, generate **4 distinct** Question-Answer examples.

Context:
"{context}"

Task:
Generate a JSON LIST containing 2 'Yes' and 2 'No' Question-Answer pairs.

Noise Requirements (CRITICAL):
- Questions MUST sound like real employees:
  - Slightly unclear, overly long, or with mild incorrect assumptions.
  - May include unnecessary details or uncertainty.
- Explanations MUST be correct but:
  - Slightly vague or overly general.
  - Avoid sharp legal wording.
- DO NOT change the legal truth of the answer.
- DO NOT invent rules not present in the context.

Field Requirements:
1. **question**: Practical question by a bank employee, written imperfectly.
2. **answer**: "Yes" or "No".
3. **citation**: **VERBATIM COPY-PASTE** from the text.
   - Copy EXACTLY as it appears.
   - Do NOT rephrase or merge sentences.
4. **explanation**:
   - One sentence.
   - General principle, slightly non-specific.
   - Do NOT refer to the text or document.
5. **source_details**: Format: "Header, Section X" (or just Header).

Output format:
Return ONLY a valid JSON LIST of objects.
"""


prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
chain = prompt | llm | StrOutputParser()

def clean_json_string(s):
    match = re.search(r'\[.*\]', s, re.DOTALL)
    if match: return match.group(0)
    return s

def get_doc_number(filename):
    match = re.match(r'(\d+)', filename)
    if match: return match.group(1)
    return filename

# Manually noisify the question to make it more humane.
def add_question_noise(question):
    if not question:
        return question

    if random.random() < 0.3:
        question = question.replace("?", " ??")
    if random.random() < 0.3:
        question = "So basically, " + question
    if random.random() < 0.2:
        question = question + " I just want to make sure."
    return question



def generate_synthetic_data_banker():
    print("--- Loading Chunks ---")
    chunks = load_and_split_pdfs()

    # Save Chunks

    if not chunks: 
        print("Error: No chunks found.")
        return

    # === Change: randomly select 100 chunks from all available ===
    # Using min ensures it won't crash if there are fewer than 100 chunks total
    print(f"Selecting {SAMPLE_SIZE} random chunks from {len(chunks)} available...")
    selected_chunks = random.sample(chunks, min(len(chunks), SAMPLE_SIZE))

    # ============================================================
    
    results = []
    output_file = "../../Data/RegulAItion_dataset_noise.json" # Output File
    
    print(f"\n--- Starting Run (Unified Source Field) ---")
    
    for i, chunk in enumerate(selected_chunks):
        print(f"Processing chunk {i+1}/{len(selected_chunks)}...")
        
        try:
            # Sending to GPT
            response_text = chain.invoke({"context": chunk.page_content})
            
            cleaned_response = clean_json_string(response_text)
            data_list = json.loads(cleaned_response)
            
            if isinstance(data_list, dict): data_list = [data_list]
            
            # Data from Python (file name and page)
            filename = os.path.basename(chunk.metadata.get('source', 'Unknown'))
            doc_num = get_doc_number(filename)
            page_num = chunk.metadata.get('page', 0) + 1 # 1 page based

            for item in data_list:
                # The model returns the text (title and section)
                text_location = item.get("source_details", "General")
                
                # We prepend the document and page at the beginning
                # Final result: 411-10, AML Officer, Section 14(b)
                final_source = f"{doc_num}-{page_num}, {text_location}"

                entry = {
                    "question": add_question_noise(item.get("question")),
                    "context": chunk.page_content,
                    "answer": item.get("answer"),
                    "citation": item.get("citation"),
                    "explanation": item.get("explanation"),
                    "source": final_source
                }
                results.append(entry)
            
            print(f" -> [V] Success! Added {len(data_list)} pairs.")
            
            # Continuous saving
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            
        except Exception as e:
            print(f" -> [X] Error on this chunk: {e}")

    print(f"\nDone! Final file: '{output_file}'")

if __name__ == "__main__":
    generate_synthetic_data_banker()