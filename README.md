# 🤖 Regul𝔸𝕀tions
This repository contains our course project materials for "Large Language Models for Natural Language Processing"  
designed by Dr. Sasha Apartsin.  

Here you will find our project presentations (proposal, interim, final), code, datasets, results, and visual abstract.  

## 🎯 **Project Motivation**
The banking industry is heavily regulated, and professionals often need to consult multiple complex documents to determine the legality of specific actions.  
Current tools are time-consuming, error-prone, and do not provide precise references.  
Our goal is to leverage Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to assist banking professionals by quickly identifying whether a requested action is permitted, concentrating relevant rules in a single centralized place, and generating clear explanatory answers.

## 🧩 **Problem Statement**
Banking professionals frequently face questions such as:  
"*Is it allowed to provide agricultural credit under these conditions?*"

Answering such questions requires navigating through dense regulations and cross-referencing multiple official documents. Existing systems fail to provide:
- Direct yes/no classification (whether an action is allowed).  
- Precise references to relevant regulatory documents, sections, and paragraphs.  
- Clear explanatory text supporting the answer.

Our project addresses this by building a Regulatory Banking Q&A model that takes a query and real regulatory documents as input, and outputs:
- Classification – Is the action possible or not.  
- Document reference – Specific sections and paragraphs from official rules.  
- Generated answer – Clear explanation based on the regulations.  

The model leverages RAG to retrieve relevant document chunks and is trained on a dataset containing questions, classifications, precise rules, and example answers.

## 📁 **Repository Structure**
- 📁[Presentations](https://github.com/ohadbitton1/Banking-Regulation-QA/tree/main/Presentations) – Proposal, interim, and final presentations

- 📁[Environment_dependencies](https://github.com/ohadbitton1/RegulAItion/tree/main/Environment_dependencies) - Libraries and environment settings

- 📁[Code](https://github.com/ohadbitton1/Banking-Regulation-QA/tree/main/Code) – Implementation
    - 📁[Baseline_notebooks](https://github.com/ohadbitton1/RegulAItion/tree/main/Code/Baseline_notebooks) – Notebooks for initial model experiments
    - 📁[data_generation_&_validation](https://github.com/ohadbitton1/RegulAItion/tree/main/Code/data_generation_%26_validation) - Scripts for generating and validating datasets
    - 📄[EDA.py](https://github.com/ohadbitton1/Banking-Regulation-QA/blob/main/Code/EDA.py) – Exploratory data analysis script
    - 📄[prepare_for_colab.py](https://github.com/ohadbitton1/RegulAItion/blob/main/Code/prepare_for_colab.py) – Converts the raw dataset into Train/Test JSON files for LLM fine-tuning.
    - 📄[create_inference_report.py](https://github.com/ohadbitton1/RegulAItion/blob/main/Code/create_inference_report.py) – Generates a CSV report comparing model predictions with ground-truth answers.

- 📁[Data](https://github.com/ohadbitton1/Banking-Regulation-QA/tree/main/Data) – Datasets
    - 📁[FT_datasets](https://github.com/ohadbitton1/Banking-Regulation-QA/tree/main/Data/FT_datasets) – Train and Test data sets for Fine Tuning
    - 📄[RegulAItion_dataset.json](https://github.com/ohadbitton1/Banking-Regulation-QA/blob/main/Data/RegulAItion_dataset.json) – Dataset containing questions, classifications, relevant document chunk & sections, and example answers

- 📁[Models](https://github.com/ohadbitton1/RegulAItion/tree/main/Models) - Saved model weights and configurations
    - 📁[Baseline_LoRA](https://github.com/ohadbitton1/RegulAItion/tree/main/Models/baseline_LoRA) - Pretrained LoRA model checkpoints

-  📁[Results](https://github.com/ohadbitton1/RegulAItion/tree/main/Results) – Model evaluation metrics and outputs
    - 📁[Inference_report](https://github.com/ohadbitton1/RegulAItion/tree/main/Results/Inference_report) - baseline model predictions compared to ground-truth answers

- 📁[Visuals](https://github.com/ohadbitton1/Banking-Regulation-QA/tree/main/Visuals) – Diagrams, visual abstracts, and illustrations
    - 📁[EDA](https://github.com/ohadbitton1/Banking-Regulation-QA/tree/main/Visuals/EDA) – Exploratory data analysis visualizations

- 📁[Resources](https://github.com/ohadbitton1/Banking-Regulation-QA/tree/main/Resources) – Supplementary materials and external references

## 🎓 **Team Members**
- Yossef Okropiridze
- Ohad Biton
- Michael Naftalishen
