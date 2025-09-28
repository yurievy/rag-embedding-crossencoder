import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

from retriever import raw_data, normalize_text, encode_text, search_top_k, rerank_questions
from cross_encoder import rerank_questions_cross_encoder
from rag_generation import rag_prompt, rag_generation
from formatting import format_result

# -------------------------------
# Configure logger
# -------------------------------
logger = logging.getLogger(__name__)
logger.info("Initializing RAG pipeline...")

# -------------------------------
# Global constants / flags
# -------------------------------
TOP_K_RETRIEVE = 3  # Number of top chunks/questions to retrieve

FLAGS: Dict[str, Tuple[str, Any]] = {
    "-s": ("show_Score", True),
    "-ce": ("use_cross_encoder", True),
    "-1": ("use_top_k", 1),
    "-2": ("use_top_k", 2),
    "-3": ("use_top_k", 3)
}

# -------------------------------
# Helper functions
# -------------------------------
def parse_question_flags(raw_question: List[str], flags: Dict[str, Tuple[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Parse special flags from the question and return cleaned question and updated parameters.

    Args:
        raw_question (List[str]): List of words from the user's question.
        flags (Dict[str, Tuple[str, Any]]): Mapping of flag strings to (param_name, value).

    Returns:
        Tuple[str, Dict[str, Any]]: Cleaned question and parameters dictionary.
    """
    params: Dict[str, Any] = {
        "show_Score": False,
        "use_cross_encoder": False,
        "use_top_k": 3
    }

    question_words: List[str] = []
    for word in raw_question:
        if word in flags:
            param_name, value = flags[word]
            params[param_name] = value
        else:
            question_words.append(word)

    question: str = " ".join(question_words)
    return question, params


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


# -------------------------------
# Main pipeline function
# -------------------------------
def process_question(data: Dict[str, Any]) -> str:
    """
    Process a user question: retrieve top results, optionally rerank using Cross-Encoder,
    optionally generate RAG answer, and format result as HTML.

    Args:
        data (Dict[str, Any]): Incoming JSON data containing 'question' and optional 'use_RAG'.

    Returns:
        str: Formatted HTML result for display.
    """
    # -------------------------------
    # 1. Parse input and flags
    # -------------------------------
    raw_question: List[str] = data.get("question", "").split()
    use_RAG: bool = data.get("use_RAG", False)

    question, params = parse_question_flags(raw_question, FLAGS)
    logger.info(f"Processing question: '{question}' | RAG: {use_RAG} | Params: {params}")

    # -------------------------------
    # 2. Encode question embedding
    # -------------------------------
    question_emb = encode_text([normalize_text(question)])

    # -------------------------------
    # 3. Retrieve top answers
    # -------------------------------
    top_answers = search_top_k(question_emb, top_k=TOP_K_RETRIEVE)

    # -------------------------------
    # 4. Optional reranking using Cross-Encoder
    # -------------------------------
    if params["use_cross_encoder"]:
        top_answers = rerank_questions_cross_encoder(
            top_answers, raw_data, question, top_k=params["use_top_k"]
        )
    else:
        top_answers = rerank_questions(top_answers, question_emb, top_k=params["use_top_k"])

    # -------------------------------
    # 5. Prepare links dictionary
    # -------------------------------
    links: Dict[str, Any] = {}
    for index, record in enumerate(top_answers):
        raw_record = raw_data[record["id"]]
        link = raw_record["link"]
        if link not in links:
            links[link] = (record["score"], raw_record["question"])

    # -------------------------------
    # 6. Generate RAG answer (if requested)
    # -------------------------------
    answer: str = ""
    if use_RAG:
        lines: List[str] = [
            f"Source {index}: {raw_data[record['id']]['question']}\n{raw_data[record['id']]['answer']}\n"
            for index, record in enumerate(top_answers)
        ]
        answer = rag_generation(question, "\n".join(lines), rag_prompt)

    # -------------------------------
    # 7. Format final HTML result
    # -------------------------------
    result: str = format_result(question, answer, links, params["show_Score"], use_RAG)

    # -------------------------------
    # 8. Save log
    # -------------------------------
    save_log(question, answer, use_RAG)

    return result
