import os
from collections import Counter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdfs():
    # הנתיב לתיקיית ה-data
    directory_path = "./data"
    
    print(f"--- Starting Process ---")
    print(f"Looking for PDFs in: {os.path.abspath(directory_path)}")
    
    # 1. טעינת כל הקבצים מהתיקייה
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    
    if not documents:
        print("ERROR: No documents found! Please check the 'data' folder.")
        return []

    print(f"\nSuccessfully loaded {len(documents)} pages from the PDFs.")

    # 2. הגדרת ה-Splitter (חלוקה למקטעים)
    # chunk_size=1500: גודל המאפשר הכנסת סעיף רגולטורי מלא והקשר
    # chunk_overlap=200: חפיפה למניעת איבוד מידע במעבר בין פסקאות
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=130,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # 3. ביצוע החלוקה בפועל
    chunks = text_splitter.split_documents(documents)
    
    print(f"\n------------------------------------------------")
    print(f"Total chunks created: {len(chunks)}")
    print(f"------------------------------------------------")
    
    # 4. בדיקה סטטיסטית - כמה צ'אנקים הגיעו מכל קובץ?
    # זה מוודא שכל המסמכים (310, 311 וכו') נכללו
    print("\n--- Sources Statistics (Chunks per File) ---")
    if chunks:
        # שליפת שם הקובץ מכל מקטע
        sources = [os.path.basename(chunk.metadata.get('source', 'Unknown')) for chunk in chunks]
        stats = Counter(sources)
        
        for filename, count in stats.items():
            print(f" [V] {filename}: {count} chunks")
    
    # 5. הצצה למקטע הראשון לבדיקת תקינות
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