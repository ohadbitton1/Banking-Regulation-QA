import json
import os
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# Define files and folders
# ==========================================
JSON_PATH = "../DATA/RegulAItion_dataset.json"  # Clean JSON
SCORES_CSV = "../DATA/verification_scores.csv"  # verify_ultimate scores
EDA_DIR = "EDA"
os.makedirs(EDA_DIR, exist_ok=True)
# ==========================================

def scatter_scores():
    df = pd.read_csv(SCORES_CSV)

    x = df["score_q_vs_cit"]
    y = df["score_exp_vs_cit"]
    passed = df["passed"]

    coeffs = np.polyfit(x, y, 1)
    trend = np.poly1d(coeffs)

    plt.figure(figsize=(8, 6))
    plt.scatter(x[passed], y[passed], alpha=0.6, label="Passed", color="green")
    plt.plot(x, trend(x), linewidth=2, color="blue", label="Trend Line")

    plt.xlabel("Question ↔ Citation Similarity")
    plt.ylabel("Explanation ↔ Citation Similarity")
    plt.title("Similarity Scores Scatter Plot")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(EDA_DIR, "scatter_plot.png"), dpi=300)
    plt.close()


def instances_per_file():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_numbers = []
    for item in data:
        source = item.get("source", "")
        match = re.match(r"(\d+)-\d+", source)
        if match:
            file_numbers.append(match.group(1))

    counts = Counter(file_numbers)

    # Sort numerically
    sorted_files = sorted(counts.keys(), key=int)
    sorted_counts = [counts[f] for f in sorted_files]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_files, sorted_counts, color="skyblue")
    plt.xlabel("Source File Number")
    plt.ylabel("Number of Instances")
    plt.title("Instances per Source File")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig(os.path.join(EDA_DIR, "file_source_barplot.png"), dpi=300)
    plt.close()


def instances_per_file_and_section():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    sources = [item["source"] for item in data if item.get("source")]
    counts = Counter(sources)

    # פונקציה בטוחה לסידור לפי המספר הראשי במחרוזת
    def get_sort_key(source):
        match = re.search(r"\d+", source)
        if match:
            return int(match.group(0))
        else:
            return float('inf')  # סוגר את הערכים שלא ניתן לפענח בסוף

    sorted_sources = sorted(counts.keys(), key=get_sort_key)
    sorted_counts = [counts[s] for s in sorted_sources]

    plt.figure(figsize=(12, 6))
    plt.bar(sorted_sources, sorted_counts, color="skyblue")
    plt.xticks(rotation=90)
    plt.xlabel("Source (File-Section)")
    plt.ylabel("Number of Instances")
    plt.title("Instances per File & Section")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig(os.path.join(EDA_DIR, "file_section_source_barplot.png"), dpi=300)
    plt.close()



def yes_no_distribution():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    answers = [item["answer"] for item in data if item.get("answer")]
    counts = Counter(answers)

    plt.figure(figsize=(6, 5))
    plt.bar(counts.keys(), counts.values(), color="skyblue")
    plt.xlabel("Answer")
    plt.ylabel("Number of Instances")
    plt.title("Yes / No Distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig(os.path.join(EDA_DIR, "yes_no_barplot.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    scatter_scores()
    instances_per_file()
    instances_per_file_and_section()
    yes_no_distribution()
    print("✅ EDA completed successfully")
