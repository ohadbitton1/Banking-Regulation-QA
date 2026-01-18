import os
import sys
import json
import torch
import chromadb

# Imports for Colab environment
try:
    from unsloth import FastLanguageModel
    from google.colab import drive
except ImportError:
    pass 

# ================= CONFIGURATION =================

PROJECT_ROOT = "/content/drive/MyDrive/RegulAItion"
DB_PATH = os.path.join(PROJECT_ROOT, "Data", "RAG_db_all")
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "Models", "Llama3.1_adapter")

NA_MESSAGE = "I am unable to answer this question based on the provided context. Please try to rephrase or ask something else."

# ================= 1. SETUP & MOUNT =================

def setup_environment():
    """Mounts Drive and validates project paths."""
    print("🚀 Initializing Environment...")
    if os.path.exists('/content') and not os.path.exists('/content/drive'):
        print("📂 Mounting Google Drive...")
        drive.mount('/content/drive')

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"❌ DB not found at {DB_PATH}")
    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(f"❌ Adapter not found at {ADAPTER_PATH}")
    print(f"✅ Environment Ready.")

def load_rag_system():
    """Loads ChromaDB and the Fine-Tuned Model via Unsloth."""
    print("⏳ Loading Vector Database...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name="regulations")
    
    print(f"⏳ Loading Model (Unsloth optimized)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = ADAPTER_PATH, 
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model) 
    print("✅ System Loaded Successfully!")
    return collection, model, tokenizer

# ================= 2. RETRIEVAL (BLOCK FORMATTING) =================

def retrieve_context_with_sources(collection, query, n_results=5):
    """
    Fetches chunks and formats them into numbered blocks with metadata headers.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    formatted_context = ""
    if results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            text = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            
            # Extract metadata: Support 'page_label' and 'page'
            fname = os.path.basename(meta.get('source', 'Unknown'))
            page = meta.get('page_label') or meta.get('page') or 'N/A'
            
            # Format block with clear source tag for the LLM
            formatted_context += f"--- BLOCK {i+1} ---\n"
            formatted_context += f"[Source: {fname}, Page {page}]\n"
            formatted_context += f"Content: {text}\n\n"
            
    return formatted_context

# ================= 3. GENERATION (PRECISE CITATION) =================

def generate_answer(model, tokenizer, query, context):
    """
    Generates strict JSON response with a single mapped source for the quote.
    """
    system_instruction = f"""You are a strict banking regulation assistant. 
Based strictly on the provided Context blocks, answer the User Query.

RULES:
1. If the Context contains the answer:
   - Determine if the answer is "Yes" or "No".
   - Extract a direct "quote" from the text.
   - Write a short "explanation".
   - Identify the exact "source" from the [Source: ...] tag above the block you used.
   - Format output as JSON: {{"verdict": "Yes" or "No", "quote": "...", "explanation": "...", "source": "..."}}

2. If the Context DOES NOT contain the answer:
   - Format output as JSON: {{"verdict": "N.A", "explanation": "{NA_MESSAGE}", "source": "N/A"}}

3. Do NOT output anything else. Only the JSON object.
"""

    full_prompt = f"""### Instruction:
{system_instruction}

### Input:
Context:
{context}

Query:
{query}

### Response:
"""
    
    inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256,
        use_cache=True,
        temperature=0.1, # Keep low for structured output
    )
    
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Extract JSON part
    raw_answer = generated_text.split("### Response:\n")[-1].strip()
    if "}" in raw_answer:
        raw_answer = raw_answer.split("}")[0] + "}"
        
    return raw_answer

# ================= 4. MAIN LOOP =================

def main():
    setup_environment()
    collection, model, tokenizer = load_rag_system()
    
    print("\n" + "="*50)
    print("🤖  RegulAItion RAG Agent (Enhanced Sources)")
    print("="*50)
    
    while True:
        try:
            user_query = input("\n❓ Question (or 'exit'): ")
            if user_query.lower() in ['exit', 'quit']:
                print("👋 Exiting...")
                break
            if not user_query.strip(): continue

            print(f"🔎 Retrieving info...")
            # Get block-formatted context
            context = retrieve_context_with_sources(collection, user_query)
            
            if not context:
                context = "No relevant documents found."

            print(f"🧠 Generating answer...")
            json_response_str = generate_answer(model, tokenizer, user_query, context)
            
            try:
                # Parse JSON string to dictionary
                response_data = json.loads(json_response_str)
                
                print("-" * 60)
                verdict = response_data.get('verdict', 'Unknown')
                print(f"Verdict:  {verdict}")
                
                if verdict in ['Yes', 'No']:
                    print(f"Quote:    \"{response_data.get('quote', 'N/A')}\"")
                    print(f"Explain:  {response_data.get('explanation', 'N/A')}")
                    print(f"Source:   {response_data.get('source', 'N/A')}") # Precise Source
                else:
                    print(f"Message:  {response_data.get('explanation', 'N/A')}")
                
                print("-" * 60)

            except json.JSONDecodeError:
                print("⚠️ Warning: Output format error.")
                print(f"Raw Output: {json_response_str}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()