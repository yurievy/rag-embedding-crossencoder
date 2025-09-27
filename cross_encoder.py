from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)

logger.info("Initializing Cross-Encoder reranker...")
reranker = CrossEncoder(
    'jinaai/jina-reranker-v2-base-multilingual',
    device='cpu',
    trust_remote_code=True
)
logger.info("Cross-Encoder reranker successfully loaded.")

def rerank_questions_cross_encoder(top_chunk_results, raw_data, question, top_k=3):
    
    # Build pairs: (question, candidate answer)
    pairs = [
        (question, raw_data[item["id"]]['answer'])
        for item in top_chunk_results
    ]

    # Get scores from the cross-encoder
    scores = reranker.predict(pairs)

    # Combine items with their scores
    reranked = [
        {"id": item["id"], "score": score}
        for item, score in zip(top_chunk_results, scores)
    ]
    
    # Sort by score (descending)
    reranked.sort(key=lambda x: x['score'], reverse=True)

    # Return top-k results
    return reranked[:top_k]