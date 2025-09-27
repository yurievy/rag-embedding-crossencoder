from sentence_transformers import CrossEncoder
import logging
from typing import List, Dict, Any

# -------------------------------
# Constants
# -------------------------------
CROSS_ENCODER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
DEVICE = "cpu"

# -------------------------------
# Configure logger
# -------------------------------
logger = logging.getLogger(__name__)
logger.info("Initializing Cross-Encoder reranker...")

# -------------------------------
# Initialize CrossEncoder model
# -------------------------------
reranker = CrossEncoder(
    CROSS_ENCODER_MODEL,
    device=DEVICE,
    trust_remote_code=True
)
logger.info(f"Cross-Encoder reranker '{CROSS_ENCODER_MODEL}' successfully loaded.")


# -------------------------------
# Function to rerank candidate answers
# -------------------------------
def rerank_questions_cross_encoder(
    top_chunk_results: List[Dict[str, Any]],
    raw_data: Dict[str, Dict[str, str]],
    question: str,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Rerank candidate answers using a Cross-Encoder.

    Args:
        top_chunk_results (List[Dict]): List of candidate chunks with at least 'id' key.
        raw_data (Dict): Dictionary mapping chunk IDs to data containing 'answer'.
        question (str): The question to compare against candidate answers.
        top_k (int): Number of top results to return.

    Returns:
        List[Dict]: Top-k chunks with their reranked scores.
    """
    # Build pairs: (question, candidate answer)
    pairs = [(question, raw_data[item["id"]]['answer']) for item in top_chunk_results]

    # Compute relevance scores using the cross-encoder
    scores = reranker.predict(pairs)

    # Combine each item with its score
    reranked = [
        {"id": item["id"], "score": score}
        for item, score in zip(top_chunk_results, scores)
    ]

    # Sort by score descending
    reranked.sort(key=lambda x: x['score'], reverse=True)

    # Return top-k results
    return reranked[:top_k]
