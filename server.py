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

app = Flask(__name__)

winsound.Beep(600, 350)

# Write results to the log
def save_log(question: str, answer: str, use_RAG: bool, log_file: str = "q_log.txt"):
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
    
    question = data.get("question", "")
    use_RAG = data.get("use_RAG", False)

    show_Score = " -s" in question
    use_cross_encoder = " -ce" in question
    
    if show_Score:
        question = question.replace(" -s", "").strip()
    if use_cross_encoder:
        question = question.replace(" -ce", "").strip()
    
    use_top_k = 3
    if " -3" in question:
        question = question.replace(" -3", "").strip()
    if " -2" in question:
        question = question.replace(" -2", "").strip()
        use_top_k = 2
    if " -1" in question:
        question = question.replace(" -1", "").strip()
        use_top_k = 1

    question_norm = normalize_text(question)
    question_emb = encode_text([question_norm])

    top_answers = search_top_k(question_emb, top_k=3)
    
    if use_cross_encoder:
        top_answers = rerank_questions_cross_encoder(top_answers, raw_data, question, top_k=use_top_k)
    else:
        top_answers = rerank_questions(top_answers, question_emb, top_k=use_top_k)

    lines = []
    links = []
    
    for index, record in enumerate(top_answers):
        
        score = record["score"]
        
        raw_record = raw_data[record['id']]
        raw_query = raw_record['question']
        
        lines.append(f"Source {index}: {raw_query}")
        lines.append(f"{raw_record['answer']}")
        lines.append("")
        
        link_html = f"<p style='margin-left:4em'>{str(score) + " " if show_Score else ""}<a href='{raw_record['link']}' target='_blank'>{raw_query}</a>"
        
        if link_html not in links:
            links.append(link_html)

    retrieved = "\n".join(lines)
    all_links = "\n".join(links)

    if use_RAG:
        answer = rag_generation(question, retrieved, rag_prompt)
    else:
        answer = ""

    answer = answer.replace("[url]", "<a href='https://www.1bpm.ru' target='blank'>www.1bpm.ru</a>")
    answer = answer.replace("[mail]", "<a href='mailto:mail@1bpm.ru?subject=КонструкторБизнесПроцессов' target='blank'>mail@1bpm.ru</a>")
    answer = answer.replace("прайс-лист", "<a href='https://1bpm.ru/price_list' target='blank'>Прайс-лист</a>")

    result = '<div align="right"><b style="background-color:#f4f4f4;margin-bottom:0em;padding: 0.5em 1em;margin: 1em 0;border-radius: 10px;display: inline-block;">' + question + '</b></div>'
    result += '<div style="background-color:#fafafa;margin-bottom:3em;padding: 0.0em 1em;border-radius: 10px;display: inline-block;">'
    
    if use_RAG:
        result += f"<p style='margin-left:2em'><b>AI</b>: {answer}<br>"
        result += "<p style='margin-left:2em'>Подробнее см.:"
    else:
        result += "<p style='margin-left:2em'>Материалы по данной теме:"

    result += all_links
    result += "</div>"

    # Save log
    save_log(question, answer, use_RAG)

    return jsonify({"answer": result})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)