# FinanceBench RAG Evaluation – Embedding Model Comparison

This project evaluates various embedding models in a Retrieval-Augmented Generation (RAG) pipeline using the **FinanceBench** dataset, a benchmark for financial question answering with grounding in official documents.

## 🔍 Project Goals

- Compare the effectiveness of different embedding models in retrieving contextually relevant financial documents.
- Analyze semantic similarity between financial questions and grounding documents.
- Benchmark models using cosine similarity and identify performance trends.

## 🚀 Models Compared

- 🧠 [Cohere AI Embed V3](https://docs.cohere.com/docs/embed)
- 🌐 [Voyage AI v1](https://docs.voyageai.com/)
- 🏃‍♂️ [MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## 📊 Methodology

1. **Dataset**: Used the FinanceBench dataset (available [here](https://huggingface.co/datasets/gaussalgo/FinanceBench)).
2. **Preprocessing**: Extracted questions and corresponding grounding document content.
3. **Embedding Generation**: Generated vector embeddings using each model.
4. **Similarity Measurement**: Used cosine similarity to evaluate question-document semantic alignment.
5. **Retrieval Simulation**: Retrieved top-k relevant documents for each question and compared across models.

## 🧰 Tech Stack

- Python
- Hugging Face Datasets & Transformers
- Cohere AI SDK
- Voyage AI SDK
- SentenceTransformers
- Scikit-learn (for cosine similarity)


## 📈 Sample Output

- Cosine similarity scores for each question-document pair
- Top-k most relevant documents per question (model-wise)
- Comparative analysis graphs (optional with matplotlib)

## 📝 Results Summary

| Model             | Avg. Cosine Similarity | Notes                          |
|------------------|------------------------|-------------------------------|
| Cohere AI        | 0.82                   | Strong alignment, fast API    |
| Voyage AI        | 0.80                   | Context-rich embeddings       |
| MiniLM-L6-v2     | 0.74                   | Lightweight, decent accuracy  |

## 🧪 How to Run

```bash
pip install -r requirements.txt
python embedding_generator.py
python similarity_analysis.py

