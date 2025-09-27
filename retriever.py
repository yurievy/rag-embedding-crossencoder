import logging
import json
import string
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Load and prepare the model and embeddings ---

model_name = "intfloat/multilingual-e5-base"
raw_data_name = "data/raw_" + model_name.replace("/", "-") + ".json"
dataset_name = "data/dataset_" + model_name.replace("/", "-") + ".json"

logger.info("Initializing embedding retriever...")

with open(raw_data_name, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

model = SentenceTransformer(model_name)

logger.info("Embedding retriever successfully loaded.")

logger.info("Loading embeddings...")

with open(dataset_name, "r", encoding="utf-8") as f:
    dataset = json.load(f)

embeddings = np.array([item["embedding"] for item in dataset])

logger.info("Embeddings are loaded.")

def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def encode_text(text):
    return model.encode(text)

# --- Function to search chunks with unique questions ---
def search_top_k(question_emb, top_k=5, chunk_top_k=50):
    
    similarities = cosine_similarity(question_emb, embeddings)[0]

    # Search for relevant chunks
    best_idxs = similarities.argsort()[::-1][:chunk_top_k]

    # Group by questions
    scores_by_question = {}
    chunks_by_question = {}
    for idx in best_idxs:
        
        q_id = dataset[idx]["question_id"]
        chunk_text = dataset[idx].get("chunk_text", "")
        score = similarities[idx]

        # Store the best score
        if q_id not in scores_by_question or score > scores_by_question[q_id]:
            scores_by_question[q_id] = score

        # Store top chunks for each question
        if q_id not in chunks_by_question:
            chunks_by_question[q_id] = [(score, chunk_text)]
        else:
            chunks_by_question[q_id].append((score, chunk_text))
            # Keep a maximum of 3 best chunks
            chunks_by_question[q_id] = sorted(chunks_by_question[q_id], key=lambda x: x[0], reverse=True)[:3]

    # Sort questions
    sorted_q_ids = sorted(scores_by_question.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Compile results
    results = []
    for q_id, score in sorted_q_ids:
        results.append({
            "id": int(q_id),
            "score": score,
            "top_chunks": [chunk[:100] for s, chunk in chunks_by_question[q_id]]
        })
    
    return results

def rerank_questions(top_chunk_results, question, top_k=3):

    scores = []
    for item in top_chunk_results:
        id = item["id"]
        raw_query = raw_data[id]['answer']
        raw_embedding = model.encode([raw_query])
        score = cosine_similarity(question, raw_embedding)[0][0]
        scores.append({
            "id": id,
            "score": score
        })
    
    scores.sort(key=lambda x: x['score'], reverse=True)

    return scores[:top_k]