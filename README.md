# 🎫 AI Support Ticket Copilot with RAG

An end-to-end AI-powered customer support system that automatically classifies support tickets, predicts urgency, assigns business priority and routing, retrieves relevant support knowledge, and generates a context-aware customer response using a hosted Large Language Model.

## 🚀 Live Demo

Try the deployed Streamlit application:

https://ai-support-ticket-copilot-rag-varfbyrvtipj9phg3cissb.streamlit.app/

---

## 📌 Project Overview

Customer support teams often need to manually:

- categorize incoming tickets
- determine urgency
- assign priority
- route tickets to the correct team
- search internal support documentation
- prepare responses for customers

This project combines traditional Machine Learning, semantic search, Retrieval-Augmented Generation (RAG), and a hosted LLM to automate this workflow.

## 🧠 System Architecture

```text
Customer Support Ticket
          ↓
TF-IDF Vectorization
          ↓
LinearSVC Classification
     ↙             ↘
Category          Urgency
     ↓               ↓
     Business Rules
          ↓
Priority + Routing + Recommended Action
          ↓
SentenceTransformer Embeddings
          ↓
Semantic Knowledge Retrieval
          ↓
Top-K + Similarity Filtering
          ↓
Relevant Support Documents
          ↓
Hosted Qwen LLM
          ↓
AI Suggested Response
```

## ✨ Features

- Automatic support ticket category classification
- Urgency prediction
- Business priority assignment (P1 / P2 / P3)
- Automatic routing to the appropriate support queue
- Recommended operational action
- Semantic search over support documentation
- Top-K knowledge retrieval using cosine similarity
- Similarity-threshold filtering
- Retrieval-Augmented Generation (RAG)
- Hosted LLM response generation
- Interactive Streamlit web application

## 🤖 Machine Learning Models

### Category Classification

The category classifier predicts:

- Billing
- Bug
- Refund
- Other

The model uses:

- TF-IDF vectorization
- Unigrams and bigrams
- Linear Support Vector Classification (LinearSVC)

Test accuracy achieved approximately **76%**.

### Urgency Classification

The urgency classifier predicts:

- High
- Medium
- Low

The final model uses TF-IDF features with LinearSVC.

Test accuracy achieved approximately **68.6%**, with a macro F1 score of approximately **67.2%**.

## 📚 Retrieval-Augmented Generation

The RAG pipeline uses:

**Embedding model**

`sentence-transformers/all-MiniLM-L6-v2`

Support documents are converted into dense semantic embeddings.

For each incoming ticket:

1. The ticket is converted into an embedding.
2. Cosine similarity is calculated against the support knowledge base.
3. The most relevant documents are retrieved.
4. A similarity threshold filters weak matches.
5. Relevant documents are supplied to the language model as context.
6. The language model generates a customer-facing response.

Similarity scores represent semantic closeness between the ticket and support documents. They are not model accuracy scores.

## 🧠 Language Model

Customer responses are generated using:

`Qwen/Qwen2.5-1.5B-Instruct`

The model is accessed through a hosted Hugging Face inference provider rather than being loaded directly into the Streamlit application.

The generation prompt instructs the model to use retrieved support information and avoid inventing unsupported policies or completed actions.

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF
- LinearSVC
- Sentence Transformers
- Cosine Similarity
- Hugging Face Inference
- Qwen
- Streamlit
- GitHub

## 📊 Example Workflow

```text
Customer:
"Our production website is unavailable and customers cannot log in."

↓ Category Prediction
Bug

↓ Urgency Prediction
High

↓ Business Logic
Priority: P1
Routing Queue: Technical Support

↓ Semantic Retrieval
Technical Support - Website Outage

↓ RAG
Relevant support documentation is provided to the LLM.

↓ Output
A context-aware customer support response is generated.
```

## ⚠️ Current Limitations

This project intentionally demonstrates both the capabilities and limitations of an ML + RAG support system.

- Category classification is not perfect and may occasionally route tickets to an incorrect category.
- Training labels are broad and contain some ambiguity.
- The knowledge base is intentionally small for demonstration purposes.
- Semantic retrieval can return multiple related documents when issues overlap.
- A small hosted language model may occasionally introduce wording that is not perfectly grounded in the retrieved documentation.
- The generated response should therefore be treated as an AI-assisted recommendation rather than an automatically approved customer response.

These limitations could be addressed in a production system using higher-quality labeled data, a larger knowledge base, stronger retrieval/reranking, larger language models, and additional response validation.

## 🔮 Future Improvements

Possible future enhancements include:

- expanding the support knowledge base
- adding vector database storage
- adding document chunking for larger knowledge bases
- implementing reranking
- adding confidence-based human escalation
- improving classifier training data
- adding response-grounding validation
- adding conversation history
- adding support-agent feedback
- monitoring retrieval and generation quality

## 🎯 What This Project Demonstrates

This project demonstrates practical experience with:

- NLP text classification
- feature engineering with TF-IDF
- multi-class machine learning
- model evaluation and error analysis
- semantic embeddings
- cosine similarity
- Retrieval-Augmented Generation
- prompt design
- LLM integration
- business-rule integration
- Streamlit deployment
- end-to-end AI application development

## 👤 Author

**Sushil Pillay**
