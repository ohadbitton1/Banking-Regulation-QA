import os
import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ================= CONFIGURATION =================
# Using raw strings for Windows paths
JSON_PATH = r"..\..\Data\Chunks\Regulatory_Rules_Chunks.json"
NEW_DB_PATH = r"..\..\Data\RAG_db_legal"
LEGAL_BERT_MODEL = "nlpaueb/legal-bert-base-uncased"

def build_database_from_json():
    print(f"📂 Loading chunks from JSON: {JSON_PATH}")
    if not os.path.exists(JSON_PATH):
        print(f"❌ Error: JSON file not found at {JSON_PATH}")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. Access the 'chunks' list from the root dictionary
    chunks_list = data.get('chunks', [])
    if not chunks_list:
        print("❌ Error: No 'chunks' key found in JSON or list is empty.")
        return

    print(f"✅ Found {len(chunks_list)} chunks. Converting to Document objects...")

    # 2. Map JSON fields to Document objects
    documents = []
    for i, item in enumerate(chunks_list):
        # Extract metadata dictionary from the chunk
        meta = item.get('metadata', {})
        
        doc = Document(
            page_content=item.get('text', ''),
            metadata={
                # Cleaning the source path to show only the filename
                "source": os.path.basename(meta.get('source', 'Unknown')),
                "page_label": str(meta.get('page_label', 'N/A')),
                "title": meta.get('title', 'Unknown')
            }
        )
        documents.append(doc)

    print(f"🔄 Documents ready. Total: {len(documents)}")

    # 3. Initialize Legal-BERT (768 Dimensions)
    # Note: On local CPU this might take a few minutes
    print(f"🧠 Initializing Legal-BERT embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=LEGAL_BERT_MODEL,
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. Build and Save ChromaDB
    print(f"💾 Building ChromaDB at: {NEW_DB_PATH}")
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=NEW_DB_PATH
    )
    
    print(f"✨ Success! Legal-BERT DB built locally from JSON with $d=768$.")

if __name__ == "__main__":
    build_database_from_json()