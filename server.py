# -------------------------------
# 1 Configure logging first
# -------------------------------
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Server starting...")

# -------------------------------
# 2 Standard libraries
# -------------------------------
import os
from datetime import datetime
import winsound
from typing import Dict, Any, List

# -------------------------------
# 3 Third-party libraries
# -------------------------------
from flask import Flask, request, jsonify, render_template

# -------------------------------
# 4 Local modules
# -------------------------------
from retriever import raw_data, normalize_text, encode_text, search_top_k, rerank_questions
from cross_encoder import rerank_questions_cross_encoder
from rag_generation import rag_prompt, rag_generation
from formatting import format_result

# -------------------------------
# 5 Initialize Flask app
# -------------------------------
app = Flask(__name__)

# -------------------------------
# 6 Optional: Beep to signal startup
# -------------------------------
winsound.Beep(600, 350)

# -------------------------------
# 7 Global constants / flags
# -------------------------------
FLAGS = {
    "-s": ("show_Score", True),
    "-ce": ("use_cross_encoder", True),
    "-1": ("use_top_k", 1),
    "-2": ("use_top_k", 2),
    "-3": ("use_top_k", 3)
}

def save_log(question: str, answer: str, use_RAG: bool, log_file: str = "qa.log") -> None:
    """
    Save question-answer pair to a log file with timestamp.

    Args:
        question (str): User's question.
        answer (str): Generated answer.
        use_RAG (bool): Whether RAG was used.
        log_file (str): Path to the log file.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{now}\n[Question: {question}] RAG used: {use_RAG}\n{answer}\n\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)


@app.route("/")
def home() -> str:
    """Render the main HTML page."""
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask() -> Any:
    """
    Handle user question, search top results, optionally rerank with Cross-Encoder,
    optionally use RAG to generate an answer, and return formatted HTML.
    """
    data = request.json or {}
    
    raw_question: List[str] = data.get("question", "").split()
    use_RAG: bool = data.get("use_RAG", False)

    # Default parameters
    params: Dict[str, Any] = {
        "show_Score": False,
        "use_cross_encoder": False,
        "use_top_k": 3
    }

    # Parse flags in the question
    question_words: List[str] = []
    for word in raw_question:
        if word in FLAGS:
            param_name, value = FLAGS[word]
            params[param_name] = value
        else:
            question_words.append(word)
    question: str = " ".join(question_words)

    logger.info(f"Processing question: '{question}' | RAG: {use_RAG} | Params: {params}")

    # Encode question
    question_emb = encode_text([normalize_text(question)])

    # Retrieve top results
    top_answers = search_top_k(question_emb, top_k=3)
    
    # Optionally rerank using Cross-Encoder
    if params["use_cross_encoder"]:
        top_answers = rerank_questions_cross_encoder(top_answers, raw_data, question, top_k=params["use_top_k"])
    else:
        top_answers = rerank_questions(top_answers, question_emb, top_k=params["use_top_k"])

    # Prepare content for RAG generation
    lines: List[str] = []
    links: Dict[str, Any] = {}

    for index, record in enumerate(top_answers):
        raw_record = raw_data[record["id"]]
        raw_query = raw_record["question"]
        raw_answer = raw_record["answer"]
        link = raw_record["link"]

        lines.extend([f"Source {index}: {raw_query}", f"{raw_answer}", ""])

        # Keep unique links with scores
        if link not in links:
            links[link] = (record["score"], raw_query)

    # Generate RAG answer if needed
    answer: str = rag_generation(question, "\n".join(lines), rag_prompt) if use_RAG else ""

    # Format result into HTML
    result: str = format_result(question, answer, links, params["show_Score"], use_RAG)

    # Save log
    save_log(question, answer, use_RAG)

    return jsonify({"answer": result})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
