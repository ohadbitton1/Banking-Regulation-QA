
import os
import sys
import json
import chromadb
from chromadb.api.types import Documents, Embeddings
from unsloth import FastLanguageModel
from peft import PeftModel
from google.colab import drive
from langchain_huggingface import HuggingFaceEmbeddings

# ================= CONFIGURATION =================
PROJECT_ROOT = "/content/drive/MyDrive/RegulAItion"
DB_PATH = os.path.join(PROJECT_ROOT, "Data", "RAG_db_legal")
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "Models", "Llama3.1_adapter")

# ================= EMBEDDING ADAPTER (CHROMA COMPAT) =================
class LegalBertAdapter:
    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name="nlpaueb/legal-bert-base-uncased",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # Chroma uses this for embedding documents
    def __call__(self, input: Documents) -> Embeddings:
        return self.model.embed_documents(input)

    # Chroma calls this for query embedding
    def embed_query(self, input=None, text=None, **kwargs):
        # Handle both kwarg 'input' or 'text' or direct arg
        q = input if input is not None else text
        if q is None:
            raise ValueError("embed_query expected 'input' or 'text'.")

        if isinstance(q, list):
            return self.model.embed_documents(q)
        return self.model.embed_query(q)

# ================= SETUP =================
def setup_environment():
    print("🚀 Initializing Environment...")

    # Mount Drive if needed
    if not os.path.exists("/content/drive"):
        drive.mount("/content/drive")

    # Validate adapter path (case fallback)
    if not os.path.exists(ADAPTER_PATH):
        alt = ADAPTER_PATH.replace("Llama", "llama")
        if os.path.exists(alt):
            return DB_PATH, alt
        raise FileNotFoundError(f"❌ Adapter not found at {ADAPTER_PATH}")

    return DB_PATH, ADAPTER_PATH

# ================= LOAD SYSTEM =================
def load_system(db_path, adapter_path):
    print("⏳ Loading Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    FastLanguageModel.for_inference(model)

    print("🧠 Connecting to LegalDB (LegalBERT)...")
    embedding_function = LegalBertAdapter()

    client = chromadb.PersistentClient(path=db_path)

    cols = client.list_collections()
    if not cols:
        raise RuntimeError(f"❌ No collections found in Chroma DB at: {db_path}")

    col = client.get_collection(
        name=cols[0].name,
        embedding_function=embedding_function,
    )

    return model, tokenizer, col

# ================= RETRIEVAL =================
def retrieve(col, query, n_results=3):
    # This call now works thanks to the fixed Adapter
    res = col.query(query_texts=[query], n_results=n_results)

    context = []
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])

    if docs and docs[0]:
        for i, doc in enumerate(docs[0]):
            meta = metas[0][i] if metas and metas[0] else {}
            context.append(
                f"Source: {meta.get('source', 'UNK')} (Page {meta.get('page_label', '0')})\n"
                f"Text: {doc}"
            )

    return "\n\n".join(context)

# ================= GENERATION (PRO PROMPT) =================
def generate(model, tokenizer, query, context):
    prompt = f"""### Instruction:
You are an expert regulatory compliance assistant.
Your task is to answer the user's question based STRICTLY on the provided context.

Rules:
1. If the answer is explicitly supported by the text, return "verdict": "Yes" or "No".
2. If the answer is NOT found in the context, return "verdict": "N.A".
3. If verdict is "N.A", set "quote" to "N.A" and "source_details" to "N.A".
4. "quote" must be an EXACT copy of the relevant text segment.
5. Return ONLY a valid JSON object with keys: "verdict", "explanation", "quote", "source_details".

### Input:
Context:
{context}

Question:
{query}

### Response:
"""
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
    text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    return text.split("### Response:")[-1].strip()

# ================= OUTPUT PARSING =================
def safe_extract_json(ans: str):
    start = ans.find("{")
    end = ans.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output.")
    clean = ans[start : end + 1]
    return json.loads(clean)

# ================= MAIN =================
def main():
    try:
        db_path, adapter_path = setup_environment()
        model, tokenizer, col = load_system(db_path, adapter_path)

        print("\n✅ LEGAL SYSTEM READY (FINAL). ASKING QUESTIONS...\n")

        while True:
            q = input("❓ Question (or 'exit'): ").strip()
            if not q:
                continue
            if q.lower() in ["exit", "quit"]:
                break

            ctx = retrieve(col, q, n_results=3)
            if not ctx:
                print("❌ DB returned no documents.")
                continue

            ans = generate(model, tokenizer, q, ctx)

            try:
                data = safe_extract_json(ans)

                print("-" * 50)
                print(f"Verdict: {data.get('verdict')}")
                if data.get("verdict") != "N.A":
                    print(f"Quote:   {data.get('quote')}")
                    # המודל הונחה להשתמש ב-source_details אז זה תקין
                    print(f"Source:  {data.get('source_details')}")
                else:
                    print(f"Quote:   N.A")

                print(f"Explain: {data.get('explanation')}")
                print("-" * 50)
            except Exception as e:
                print(f"⚠️ JSON Parsing Error: {e}")
                print(f"Raw Output: {ans}")

    except Exception as e:
        print(f"❌ Critical Error: {e}")

if __name__ == "__main__":
    main()
