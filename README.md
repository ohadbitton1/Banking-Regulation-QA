# 🤖 Regul𝔸𝕀tions
This repository contains our course project materials for "Large Language Model for Natural Language Processing"  
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
-  [Presentations](https://github.com/ohadbitton1/Banking-Regulation-QA/tree/main/Presentations) – Proposal, interim, and final presentations
-  Code – Implementation of an LLM-based Regulatory Banking Q&A model
-  Data – Dataset containing questions, classifications, relevant document sections, and example answers
-  Results – Model evaluation metrics and outputs
-  Visuals – Diagrams, visual abstracts, and illustrations

## 🎓 **Team Members**
- Yossef Okropiridze
- Ohad Biton
- Michael Naftalishen
