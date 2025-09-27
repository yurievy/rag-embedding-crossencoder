# Standard libraries
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Server starting...")

import os
from datetime import datetime
import winsound

# Third-party libraries
from flask import Flask, request, jsonify, render_template

# Local modules
from retriever import raw_data, normalize_text, encode_text, search_top_k, rerank_questions
from cross_encoder import rerank_questions_cross_encoder
from rag_generation import rag_prompt, rag_generation
from formatting import format_result

app = Flask(__name__)

flags = { #REDO
    "-s": ("show_Score", True),
    "-ce": ("use_cross_encoder", True),
    "-1": ("use_top_k", 1),
    "-2": ("use_top_k", 2),
    "-3": ("use_top_k", 3)
}

winsound.Beep(600, 350)

# Write results to the log
def save_log(question: str, answer: str, use_RAG: bool, log_file: str = "qa.log"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{now}\n[{question}] {use_RAG}\n{answer}\n\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

@app.route("/")
def home():
    return render_template("index.html")  # Return HTML

@app.route("/ask", methods=["POST"])
def ask():

    data = request.json
    
    raw_question = data.get("question", "").split()
    use_RAG = data.get("use_RAG", False)

    params = {
        "show_Score": False,
        "use_cross_encoder": False,
        "use_top_k": 3
    }

    question = []
    for word in raw_question:
        if word in flags:
            param_name, value = flags[word]
            params[param_name] = value
        else:
            question.append(word)

    question = " ".join(question)

    question_emb = encode_text([normalize_text(question)])

    top_answers = search_top_k(question_emb, top_k=3)
    
    if params["use_cross_encoder"]:
        top_answers = rerank_questions_cross_encoder(top_answers, raw_data, question, top_k=params["use_top_k"])
    else:
        top_answers = rerank_questions(top_answers, question_emb, top_k=params["use_top_k"])

    lines = []
    links = {}
    
    for index, record in enumerate(top_answers):
        
        raw_record = raw_data[record['id']]
        raw_query = raw_record['question']
        raw_answer = raw_record['answer']
        link = raw_record['link']

        lines.append(f"Source {index}: {raw_query}")
        lines.append(f"{raw_answer}")
        lines.append("")

        if link not in links:
            links[link] = (record["score"], raw_query) 
        
    answer = rag_generation(question, "\n".join(lines), rag_prompt) if use_RAG else ""
    result = format_result(question, answer, links, params["show_Score"], use_RAG)

    # Save log
    save_log(question, answer, use_RAG)

    return jsonify({"answer": result})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)