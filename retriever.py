import logging
import json
import string
import re
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# -------------------------------
# Load model and dataset
# -------------------------------
model_name = "intfloat/multilingual-e5-base"
raw_data_path = f"data/raw_{model_name.replace('/', '-')}.json"
dataset_path = f"data/dataset_{model_name.replace('/', '-')}.json"

logger.info("Initializing embedding retriever...")

# Load raw data
with open(raw_data_path, "r", encoding="utf-8") as f:
    raw_data: Dict[str, Dict[str, Any]] = json.load(f)

# Load SentenceTransformer model
model = SentenceTransformer(model_name)

logger.info("Embedding retriever successfully loaded.")

# Load dataset with embeddings
logger.info("Loading embeddings...")
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset: List[Dict[str, Any]] = json.load(f)

embeddings = np.array([item["embedding"] for item in dataset])
logger.info("Embeddings are loaded.")


# -------------------------------
# Text preprocessing
# -------------------------------
def normalize_text(text: str) -> str:
    """
    Lowercase, remove punctuation, normalize whitespace.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def encode_text(texts: List[str]) -> np.ndarray:
    """
    Encode a list of texts using the sentence transformer model.
    """
    return model.encode(texts)


# -------------------------------
# Search top-k chunks with unique questions
# -------------------------------
def search_top_k(question_emb: np.ndarray, top_k: int = 5, chunk_top_k: int = 50) -> List[Dict[str, Any]]:
    """
    Search for top-k relevant questions and their top chunks.

    Args:
        question_emb (np.ndarray): Encoded question embedding.
        top_k (int): Number of top questions to return.
        chunk_top_k (int): Number of top chunks to consider per search.

    Returns:
        List[Dict]: List of top questions with scores and top chunks.
    """
    similarities = cosine_similarity(question_emb, embeddings)[0]

    # Get top chunk indices
    best_idxs = similarities.argsort()[::-1][:chunk_top_k]

    # Aggregate scores and chunks by question
    scores_by_question: Dict[str, float] = {}
    chunks_by_question: Dict[str, List[tuple]] = {}

    for idx in best_idxs:
        q_id = dataset[idx]["question_id"]
        chunk_text = dataset[idx].get("chunk_text", "")
        score = similarities[idx]

        # Keep the best score per question
        scores_by_question[q_id] = max(score, scores_by_question.get(q_id, 0.0))

        # Store top chunks per question (max 3)
        chunks_by_question.setdefault(q_id, []).append((score, chunk_text))
        chunks_by_question[q_id] = sorted(chunks_by_question[q_id], key=lambda x: x[0], reverse=True)[:3]

    # Select top questions
    sorted_q_ids = sorted(scores_by_question.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Prepare results
    results: List[Dict[str, Any]] = []
    for q_id, score in sorted_q_ids:
        results.append({
            "id": int(q_id),
            "score": score,
            "top_chunks": [chunk[:100] for s, chunk in chunks_by_question[q_id]]
        })

    return results


# -------------------------------
# Rerank top chunks using question similarity
# -------------------------------
def rerank_questions(top_chunk_results: List[Dict[str, Any]], question_emb: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Rerank chunks based on cosine similarity with the encoded question.

    Args:
        top_chunk_results (List[Dict]): Top chunks to rerank.
        question_emb (np.ndarray): Encoded question embedding.
        top_k (int): Number of top results to return.

    Returns:
        List[Dict]: Top-k reranked chunks with scores.
    """
    scores: List[Dict[str, float]] = []

    for item in top_chunk_results:
        chunk_id = item["id"]
        chunk_text = raw_data[chunk_id]['answer']
        chunk_emb = model.encode([chunk_text])
        score = cosine_similarity(question_emb, chunk_emb)[0][0]
        scores.append({"id": chunk_id, "score": score})

    # Sort by descending score and return top-k
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_k]
