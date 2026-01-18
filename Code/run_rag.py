import os
import json
import torch
import argparse
import pandas as pd
import chromadb
import shutil
from unsloth import FastLanguageModel
from langchain_huggingface import HuggingFaceEmbeddings

# ================= ARGUMENT PARSER =================

parser = argparse.ArgumentParser(description="Run RAG with local adapter copy mechanism.")
parser.add_argument("--adapter", type=str, required=True, help="Path to the adapter in Drive")
parser.add_argument("--db_path", type=str, required=True, help="Path to ChromaDB")
parser.add_argument("--db_type", type=str, choices=["minilm", "legalbert"], required=True, help="Embedding type")
parser.add_argument("--input", type=str, required=True, help="Path to input JSON")
args = parser.parse_args()

# ================= HELPER: BASE MODEL DETECTION =================

def get_base_model_name(adapter_path):
    path_lower = adapter_path.lower()
    if "saul" in path_lower or "mistral" in path_lower:
        print("🕵️ Detected Saul/Mistral Adapter.")
        return "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    elif "llama" in path_lower:
        print("🕵️ Detected Llama 3.1 Adapter.")
        return "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    else:
        print("⚠️ Unknown type. Defaulting to Llama 3.1.")
        return "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

# ================= SYSTEM LOAD =================

def load_system():
    drive_path = os.path.abspath(args.adapter)
    
    # 1. העתקת האדפטר לדיסק המקומי (הפתרון לבעיית ה-Repo ID)
    local_temp_path = "/content/temp_adapter_files"
    
    print(f"🚀 Copying adapter from Drive to local disk...")
    print(f"   Src: {drive_path}")
    print(f"   Dst: {local_temp_path}")
    
    # ניקוי שאריות והעתקה
    if os.path.exists(local_temp_path):
        shutil.rmtree(local_temp_path)
    
    try:
        shutil.copytree(drive_path, local_temp_path)
        print("✅ Copy successful! Using local path.")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to copy adapter: {e}")

    # 2. טעינת מודל הבסיס (Base Model)
    base_model = get_base_model_name(drive_path)
    print(f"⏳ Loading Base Model: {base_model}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model,
        max_seq_length = 2048,
        load_in_4bit = True,
    )

    # 3. הלבשת האדפטר מהעותק המקומי (עוקף את שגיאת ה-Drive)
    print(f"🔌 Attaching Local Adapter from: {local_temp_path}")
    model.load_adapter(local_temp_path)
    FastLanguageModel.for_inference(model)

    # 4. הגדרת Embeddings
    print(f"🧠 Loading Embeddings: {args.db_type}")
    if args.db_type == "legalbert":
        embed_fn = HuggingFaceEmbeddings(
            model_name="nlpaueb/legal-bert-base-uncased",
            model_kwargs={'device': 'cuda'}
        )
    else:
        embed_fn = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'}
        )

    # 5. טעינת DB
    print(f"🗄️ Loading ChromaDB from: {args.db_path}")
    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_collection(name="regulations")
    
    return collection, embed_fn, model, tokenizer

# ================= RAG LOGIC =================

def run_rag_inference(collection, embed_fn, model, tokenizer, question):
    # Retrieval
    query_vector = embed_fn.embed_query(question)
    results = collection.query(query_embeddings=[query_vector], n_results=5)
    
    context = ""
    if results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            source_name = os.path.basename(meta.get('source', 'Unknown'))
            page = meta.get('page_label') or meta.get('page') or 'N/A'
            context += f"--- BLOCK {i+1} ---\n[Source: {source_name}, Page {page}]\n{doc}\n\n"

    # Generation
    prompt = f"""### Instruction:
Answer based ONLY on the Context blocks. If not found, verdict is 'N.A'.
Return JSON: {{"verdict": "Yes/No/N.A", "quote": "...", "explanation": "...", "source": "..."}}

### Input:
Context:
{context}

Query:
{question}

### Response:
"""
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        temperature=0.1,
        use_cache=True
    )
    
    ans = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return ans.split("### Response:\n")[-1].strip()

# ================= MAIN EXECUTION =================

def main():
    try:
        collection, embed_fn, model, tokenizer = load_system()
    except Exception as e:
        print(f"❌ Critical System Error: {e}")
        return

    print(f"📂 Reading Input: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        test_set = json.load(f)

    all_results = []
    print(f"🚀 Starting Inference on {len(test_set)} items...")

    for i, item in enumerate(test_set):
        print(f"[{i+1}/{len(test_set)}] Q: {item['question'][:40]}...")
        raw_response = run_rag_inference(collection, embed_fn, model, tokenizer, item['question'])
        
        try:
            start = raw_response.find("{")
            end = raw_response.rfind("}")
            if start != -1 and end != -1:
                res_data = json.loads(raw_response[start:end+1])
            else:
                res_data = {"verdict": "ERROR", "explanation": "No JSON found"}
        except:
            res_data = {"verdict": "ERROR", "explanation": "JSON parsing failed"}

        all_results.append({
            "adapter": os.path.basename(args.adapter),
            "db_type": args.db_type,
            "question": item['question'],
            "expected_verdict": item['expected_verdict'],
            "actual_verdict": res_data.get('verdict', 'N/A'),
            "explanation": res_data.get('explanation', ''),
            "source": res_data.get('source', ''),
            "quote": res_data.get('quote', '')
        })

    # Save Results
    df = pd.DataFrame(all_results)
    out_name = f"results_{args.db_type}_{os.path.basename(args.adapter)}.csv"
    out_path = os.path.join(os.path.dirname(args.input), out_name)
    df.to_csv(out_path, index=False)
    print(f"\n✅ Done! Results saved to: {out_path}")

if __name__ == "__main__":
    main()