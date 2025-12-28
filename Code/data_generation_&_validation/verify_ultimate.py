import json
import re
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
#       Settings (single file only)
#  ==========================================
TARGET_FILE = "RegulAItion_dataset.json"  # Both input and output
# ==========================================

# Accuracy thresholds
THRESH_Q_VS_CIT = 0.35
THRESH_EXP_VS_CIT = 0.45
import json
import re
import os
import csv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
#        Settings
# ==========================================
TARGET_FILE = "../DATA/RegulAItion_dataset.json"   # Input + output (if working on the DATA)
SCORES_FILE = os.path.join("..", "DATA", "verification_scores.csv")  # Output for EDA
# ==========================================

THRESH_Q_VS_CIT = 0.35
THRESH_EXP_VS_CIT = 0.45


def tokenize(text):
    if not text:
        return []
    text = text.lower()
    return re.findall(r'\w+', text)


def is_subsequence(citation_text, context_text):
    citation_words = tokenize(citation_text)
    context_words = tokenize(context_text)
    if not citation_words:
        return False

    iter_context = iter(context_words)
    for word in citation_words:
        if word not in iter_context:
            return False
    return True


def filter_and_overwrite():
    print(f"--- Starting In-Place Cleanup on '{TARGET_FILE}' ---")

    if not os.path.exists(TARGET_FILE):
        print(f"Error: File '{TARGET_FILE}' not found.")
        return

    print("Loading AI model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    with open(TARGET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    approved_data = []
    discarded_count = 0
    scores_log = []

    print(f"\nProcessing {len(data)} items...\n")

    for i, item in enumerate(data):
        if (i + 1) % 10 == 0:
            print(f"Checking item {i+1}/{len(data)}...", end="\r")

        context = item.get('context', '')
        citation = item.get('citation', '')
        question = item.get('question', '')
        explanation = item.get('explanation', '')

        is_citation_valid = is_subsequence(citation, context)

        embeddings = model.encode([question, citation, explanation])
        score_q_cit = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        score_exp_cit = cosine_similarity([embeddings[2]], [embeddings[1]])[0][0]

        pass_q_cit = score_q_cit > THRESH_Q_VS_CIT
        pass_exp_cit = score_exp_cit > THRESH_EXP_VS_CIT

        passed = is_citation_valid and pass_q_cit and pass_exp_cit

        scores_log.append({
            "item_index": i + 1,
            "score_q_vs_cit": score_q_cit,
            "score_exp_vs_cit": score_exp_cit,
            "passed": passed
        })

        if passed:
            approved_data.append(item)
        else:
            discarded_count += 1
            print(f"\n🗑️ Item #{i+1} DELETED permanently:")
            if not is_citation_valid:
                print("   - Citation mismatch")
            if not pass_q_cit:
                print("   - Irrelevant Question")
            if not pass_exp_cit:
                print("   - Bad Explanation")
            print("-" * 30)

    # Overwrite with clean JSON
    if approved_data:
        with open(TARGET_FILE, 'w', encoding='utf-8') as f:
            json.dump(approved_data, f, indent=4, ensure_ascii=False)

        print(f"\n✅ '{TARGET_FILE}' overwritten successfully")
        print(f"   Original size: {len(data)}")
        print(f"   New size:      {len(approved_data)}")
        print(f"   Deleted:       {discarded_count}")
    else:
        print("\n⚠️ Safety Stop: No data survived. File NOT overwritten.")
        return

    # Ensure the folder exists before writing the CSV
    os.makedirs(os.path.dirname(SCORES_FILE), exist_ok=True)

    # Write CSV of scores for EDA
    with open(SCORES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["item_index", "score_q_vs_cit", "score_exp_vs_cit", "passed"]
        )
        writer.writeheader()
        writer.writerows(scores_log)

    print(f"📊 Scores saved to '{SCORES_FILE}'")
    print("All done!")


if __name__ == "__main__":
    filter_and_overwrite()

def tokenize(text):
    if not text: return []
    text = text.lower()
    return re.findall(r'\w+', text)

def is_subsequence(citation_text, context_text):
    citation_words = tokenize(citation_text)
    context_words = tokenize(context_text)
    if not citation_words: return False
    iter_context = iter(context_words)
    for word in citation_words:
        if word not in iter_context:
            return False
    return True

def filter_and_overwrite():
    print(f"--- Starting In-Place Cleanup on '{TARGET_FILE}' ---")
    
    if not os.path.exists(TARGET_FILE):
        print(f"Error: File '{TARGET_FILE}' not found.")
        return

    print("Loading AI model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 1. Load data to memory
    with open(TARGET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    approved_data = []
    discarded_count = 0
    
    print(f"\nProcessing {len(data)} items...\n")

    for i, item in enumerate(data):
        if (i+1) % 10 == 0:
            print(f"Checking item {i+1}/{len(data)}...", end="\r")

        # Data preparations
        context = item.get('context', '')
        citation = item.get('citation', '')
        question = item.get('question', '')
        explanation = item.get('explanation', '')

        # Validation
        is_citation_valid = is_subsequence(citation, context)
        
        embeddings = model.encode([question, citation, explanation])
        score_q_cit = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        score_exp_cit = cosine_similarity([embeddings[2]], [embeddings[1]])[0][0]
        
        pass_q_cit = score_q_cit > THRESH_Q_VS_CIT
        pass_exp_cit = score_exp_cit > THRESH_EXP_VS_CIT

        # Filtering
        if is_citation_valid and pass_q_cit and pass_exp_cit:
            approved_data.append(item)
        else:
            discarded_count += 1
            print(f"\n🗑️ Item #{i+1} DELETED permanently:")
            if not is_citation_valid: print(f"   - Citation mismatch")
            if not pass_q_cit: print(f"   - Irrelevant Question")
            if not pass_exp_cit: print(f"   - Bad Explanation")
            print("-" * 30)

    # 2. Overwrite the original file with clean data only
    if len(approved_data) > 0:
        print(f"\n\n--- Overwriting File ---")
        with open(TARGET_FILE, 'w', encoding='utf-8') as f:
            json.dump(approved_data, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Success! '{TARGET_FILE}' has been updated.")
        print(f"   Original size: {len(data)}")
        print(f"   New size:      {len(approved_data)}")
        print(f"   Deleted:       {discarded_count}")
    else:
        print("\n⚠️ Safety Stop: No data survived the filter. File was NOT overwritten.")

if __name__ == "__main__":
    filter_and_overwrite()
