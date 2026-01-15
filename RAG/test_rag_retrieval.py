import chromadb
from sentence_transformers import SentenceTransformer
import os 

# ==========================================
#              CONFIGURATIONS
# ==========================================
DB_PATH = os.path.join(os.path.dirname(__file__), "banking_rag_db")
COLLECTION_NAME = "regulations"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def query_rag(user_question, n_results=3):
    print(f"\n🔎 Querying: '{user_question}'")
    
    # 1. Connect to the existing database
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    # 2. Load the model
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # 3. Convert question to vector
    query_vector = model.encode([user_question]).tolist()
    
    # 4. Search in DB
    results = collection.query(
        query_embeddings=query_vector,
        n_results=n_results
    )
    
    # 5. Display Results
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    print("-" * 50)
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        print(f"Result #{i+1} (Score: {dist:.4f})")
        print(f"   Source: {meta['source']} (Page: {meta['page_label']})")
        print(f"   Excerpt: \"{doc[:150]}...\"") # Print only first 150 chars
        print("-" * 50)

if __name__ == "__main__":
    # Test 1: Question about Board of Directors responsibilities
    query_rag("What are the responsibilities of the board of directors regarding risk management?")
    
    # Test 2: Question about the Chief Risk Officer (CRO)
    query_rag("Who appoints the Chief Risk Officer and what is their status?")