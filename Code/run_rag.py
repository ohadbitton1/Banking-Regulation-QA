import os
import json
import torch
import argparse
import pandas as pd
import chromadb
from unsloth import FastLanguageModel
from langchain_huggingface import HuggingFaceEmbeddings

# ================= ARGUMENT PARSER =================

parser = argparse.ArgumentParser(description="Run RAG tests with flexible DB and Model selection.")
# Model adapter selection
parser.add_argument("--adapter", type=str, required=True, help="Path to the model adapter (Llama or SaulLM)")
# DB Path selection
parser.add_argument("--db_path", type=str, required=True, help="Path to the ChromaDB directory")
# DB Type selection (Determines which embedding model to use for the query)
parser.add_argument("--db_type", type=str, choices=["minilm", "legalbert"], required=True, 
                    help="Type of embeddings used in the DB: 'minilm' (384d) or 'legalbert' (768d)")
# Input/Output paths
parser.add_argument("--input", type=str, default="/content/drive/MyDrive/RegulAItion/Data/golden_test_set.json")
args = parser.parse_args()

# ================= SYSTEM LOAD =================

def load_system():
    # 1. Load the LLM Adapter
    print(f"⏳ Loading LLM Adapter: {os.path.basename(args.adapter)}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.adapter,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    # 2. Setup Embedding Function for Retrieval
    # Important: The query MUST be embedded with the same model used to build the DB
    if args.db_type == "legalbert":
        print("🧠 Using Legal-BERT Embeddings (768d)")
        embed_fn = HuggingFaceEmbeddings(
            model_name="nlpaueb/legal-bert-base-uncased",
            model_kwargs={'device': 'cuda'}
        )
    else:
        print("🧠 Using default MiniLM Embeddings (384d)")
        # If using standard Chroma/Langchain MiniLM, we can let Chroma handle it 
        # or load it explicitly for consistency
        embed_fn = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'}
        )

    # 3. Load Vector DB
    print(f"⏳ Connecting to DB at: {args.db_path}")
    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_collection(name="regulations")
    
    return collection, embed_fn, model, tokenizer

# ================= RAG LOGIC =================

def run_rag_inference(collection, embed_fn, model, tokenizer, question):
    # 1. Embed the query and retrieve
    # Convert query text to vector using the selected embedding model
    query_vector = embed_fn.embed_query(question)
    results = collection.query(query_embeddings=[query_vector], n_results=5)
    
    context = ""
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        context += f"--- BLOCK {i+1} ---\n[Source: {os.path.basename(meta['source'])}, Page {meta.get('page_label', 'N/A')}]\n{doc}\n\n"

    # 2. Generate Answer
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
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
    ans = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("### Response:\n")[-1].strip()
    return ans

# ================= MAIN EXECUTION =================

def main():
    collection, embed_fn, model, tokenizer = load_system()
    
    with open(args.input, 'r', encoding='utf-8') as f:
        test_set = json.load(f)

    all_results = []
    print(f"🚀 Processing {len(test_set)} questions using {args.db_type}...")

    for i, item in enumerate(test_set):
        print(f"[{i+1}/{len(test_set)}] Question: {item['question'][:50]}...")
        raw_response = run_rag_inference(collection, embed_fn, model, tokenizer, item['question'])
        
        try:
            clean_json = raw_response[raw_response.find("{"):raw_response.rfind("}")+1]
            res_data = json.loads(clean_json)
        except:
            res_data = {"verdict": "ERROR", "explanation": raw_response, "source": "N/A"}

        all_results.append({
            "db_type": args.db_type,
            "adapter": os.path.basename(args.adapter),
            "question": item['question'],
            "expected": item['expected_verdict'],
            "actual": res_data.get('verdict'),
            "quote": res_data.get('quote'),
            "source": res_data.get('source')
        })

    # Save results
    df = pd.DataFrame(all_results)
    output_name = f"results_{args.db_type}_{os.path.basename(args.adapter)}.csv"
    df.to_csv(os.path.join(os.path.dirname(args.input), output_name), index=False)
    print(f"✅ Results saved to {output_name}")

if __name__ == "__main__":
    main()