import os
import json
import time
import re
import string
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# -------------------------------
# Constants / Config
# -------------------------------
AVAILABLE_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base"
]

SOURCE_FOLDER = "source"
RAW_FILENAME_TEMPLATE = "data/raw_{model}.json"
DATASET_FILENAME_TEMPLATE = "data/dataset_{model}.json"

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

HEADER_PATTERN = re.compile(r"^\[query\] ")
LINK_PATTERN = re.compile(r"^\[link\] ")

# -------------------------------
# Helper functions
# -------------------------------
def sanitize_filename(name: str) -> str:
    """Sanitize string to be a valid filename."""
    name = name.replace("/", "-")
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join(c for c in name if c in valid_chars)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into chunks of specified size with overlap."""
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - chunk_overlap
    return chunks


# -------------------------------
# Main script
# -------------------------------

# Print available models
print("Available models:")
for idx, m in enumerate(AVAILABLE_MODELS, 1):
    print(f"{idx}. {m}")

# User selects a model
while True:
    try:
        choice = int(input("Enter model number: "))
        if 1 <= choice <= len(AVAILABLE_MODELS):
            MODEL_TYPE = AVAILABLE_MODELS[choice - 1]
            break
        else:
            print(f"Enter a number between 1 and {len(AVAILABLE_MODELS)}")
    except ValueError:
        print("Enter a valid number")

print(f"Selected model: {MODEL_TYPE}")
start_time = time.time()

# Load SentenceTransformer model
model = SentenceTransformer(MODEL_TYPE)

# Read source files
lines: List[str] = []
for filename in os.listdir(SOURCE_FOLDER):
    file_path = os.path.join(SOURCE_FOLDER, filename)
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines.extend([line.strip() for line in f])

# Parse into question-answer pairs
pairs: List[Dict[str, str]] = []
current_question: str = None
current_link: str = None
current_answer: List[str] = []
index = 0

for line in lines:
    line = line.strip()
    if HEADER_PATTERN.match(line):
        if current_question is not None:
            pairs.append({
                "id": index,
                "question": current_question,
                "link": current_link,
                "answer": "\n".join(current_answer)
            })
            index += 1
        current_question = line.replace("[query] ", "")
        current_link = None
        current_answer = []
    elif LINK_PATTERN.match(line):
        current_link = line.replace("[link]", "")
    elif line:
        line = line.replace("[passage]", "")
        if line:
            current_answer.append(line)

# Add last pair
if current_question is not None:
    pairs.append({
        "id": index,
        "question": current_question,
        "link": current_link,
        "answer": "\n".join(current_answer)
    })

print(f"Found {len(pairs)} question-answer pairs")

# Save raw pairs JSON
RAW_FILENAME = RAW_FILENAME_TEMPLATE.format(model=sanitize_filename(MODEL_TYPE))
with open(RAW_FILENAME, "w", encoding="utf-8") as f:
    json.dump(pairs, f, ensure_ascii=False, indent=4)

print(f"Saved raw data to {RAW_FILENAME}")

# Generate embeddings for chunks
dataset_chunks: List[Dict[str, any]] = []
for i, pair in enumerate(pairs):
    question_id = f"{i:04d}"
    answer_chunks = chunk_text(pair["answer"])

    for chunk in answer_chunks:
        text_for_embedding = f"{pair['question']} {chunk}"
        embedding_vector = model.encode([text_for_embedding])[0].tolist()
        dataset_chunks.append({
            "question_id": question_id,
            "chunk_text": chunk,
            "embedding": embedding_vector
        })

# Save dataset JSON
DATASET_FILENAME = DATASET_FILENAME_TEMPLATE.format(model=sanitize_filename(MODEL_TYPE))
with open(DATASET_FILENAME, "w", encoding="utf-8") as f:
    json.dump(dataset_chunks, f, ensure_ascii=False, indent=2)

elapsed = time.time() - start_time
print(f"Embeddings saved to {DATASET_FILENAME}")
print(f"Elapsed time: {elapsed:.2f} seconds")

input("Press Enter to exit...")
