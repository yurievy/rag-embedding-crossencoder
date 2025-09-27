import json
import html
from typing import Dict, Tuple

# Load replacement dictionary
with open("data/replace.json", "r", encoding="utf-8") as f:
    replace_dict: Dict[str, str] = json.load(f)

# Load static strings
with open("data/strings.json", "r", encoding="utf-8") as f:
    strings: Dict[str, str] = json.load(f)


def format_result(
    question: str,
    answer: str,
    links: Dict[str, Tuple[float, str]],
    show_Score: bool,
    use_RAG: bool
) -> str:
    """
    Format question, answer, and links into a safe HTML block.

    Args:
        question (str): User's question.
        answer (str): AI-generated answer.
        links (dict): Dictionary of {link: (score, query)}.
        show_Score (bool): Whether to display scores next to links.
        use_RAG (bool): Whether to display RAG-specific formatting.

    Returns:
        str: HTML-formatted result.
    """
    # Escape question and answer for safe HTML
    question_html = html.escape(question)
    answer_html = html.escape(answer)

    # Apply replacements in answer
    for key, value in replace_dict.items():
        answer_html = answer_html.replace(key, value)

    # Build links HTML
    links_html_list = []
    for link, (score, query) in links.items():
        score_text = f"{score} " if show_Score else ""
        link_html = f"<p style='margin-left:4em'>{score_text}<a href='{html.escape(link)}' target='_blank'>{html.escape(query)}</a></p>"
        links_html_list.append(link_html)
    all_links_html = "\n".join(links_html_list)

    # Build the final HTML
    result = (
        f"<div align='right'>"
        f"<b style='background-color:#f4f4f4;margin-bottom:0em;padding:0.5em 1em;"
        f"margin:1em 0;border-radius:10px;display:inline-block;'>{question_html}</b>"
        f"</div>"
        f"<div style='background-color:#fafafa;margin-bottom:3em;padding:0.0em 1em;"
        f"border-radius:10px;display:inline-block;'>"
    )

    if use_RAG:
        result += f"<p style='margin-left:2em'><b>AI</b>: {answer_html}<br>"
        result += f"<p style='margin-left:2em'>{strings['rag_links_header']}:"
    else:
        result += f"<p style='margin-left:2em'>{strings['links_header']}:"

    result += all_links_html
    result += "</div>"

    return result
