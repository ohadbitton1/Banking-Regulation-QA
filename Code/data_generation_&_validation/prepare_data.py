import os
from collections import Counter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdfs():

    # Path to the data folder
    directory_path = "../documents" # Regulation Documents pdf files path
    
    print(f"--- Starting Process ---")
    print(f"Looking for PDFs in: {os.path.abspath(directory_path)}")
    
    # 1. Load all files from the folder
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    
    if not documents:
        print("ERROR: No documents found! Please check the 'data' folder.")
        return []

    print(f"\nSuccessfully loaded {len(documents)} pages from the PDFs.")

    # 2. Set up the Splitter (divide into segments)
    # chunk_size=1500: size that allows including a full regulatory section and context
    # chunk_overlap=200: overlap to prevent losing information between paragraphs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=130,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # 3. Perform the actual splitting
    chunks = text_splitter.split_documents(documents)
    
    print(f"\n------------------------------------------------")
    print(f"Total chunks created: {len(chunks)}")
    print(f"------------------------------------------------")
    
    # 4. Statistical check – how many chunks were generated from each file?
    # This ensures all documents (310, 311, etc.) are included
    print("\n--- Sources Statistics (Chunks per File) ---")
    if chunks:
        # Extract the file name from each chunk
        sources = [os.path.basename(chunk.metadata.get('source', 'Unknown')) for chunk in chunks]
        stats = Counter(sources)
        
        for filename, count in stats.items():
            print(f" [V] {filename}: {count} chunks")
    
    # 5. Preview the first chunk to check correctness
    if chunks:
        print("\n--- Preview of the first chunk ---")
        first_chunk = chunks[0]
        source_name = os.path.basename(first_chunk.metadata.get('source', 'Unknown'))
        print(f"Source: {source_name}")
        print("Content (first 500 chars):")
        print(first_chunk.page_content[:500])
        print("...")
        
    return chunks

if __name__ == "__main__":
    final_chunks = load_and_split_pdfs()