import json

with open("data/replace.json", "r", encoding="utf-8") as f:
    replace_dict = json.load(f)

with open("data/strings.json", "r", encoding="utf-8") as f:
    strings = json.load(f)

def format_result(question, answer, links, show_Score, use_RAG):
    
    all_links_html = []
    
    for link, (score, query) in links.items():
        link_html = f"<p style='margin-left:4em'>{str(score) + ' ' if show_Score else ''}<a href='{link}' target='_blank'>{query}</a></p>"
        all_links_html.append(link_html)
    
    all_links = "\n".join(all_links_html)

    for key, value in replace_dict.items():
        answer = answer.replace(key, value)

    result = '<div align="right"><b style="background-color:#f4f4f4;margin-bottom:0em;padding: 0.5em 1em;margin: 1em 0;border-radius: 10px;display: inline-block;">' + question + '</b></div>'
    result += '<div style="background-color:#fafafa;margin-bottom:3em;padding: 0.0em 1em;border-radius: 10px;display: inline-block;">'
    
    if use_RAG:
        result += f"<p style='margin-left:2em'><b>AI</b>: {answer}<br>"
        result += f"<p style='margin-left:2em'>{strings['rag_links_header']}:"
    else:
        result += f"<p style='margin-left:2em'>{strings['links_header']}:"

    result += all_links
    result += "</div>"

    return result