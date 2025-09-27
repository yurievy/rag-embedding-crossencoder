from llama_cpp import Llama
import os
import logging

logger = logging.getLogger(__name__)

logger.info("Initializing RAG...")

model_name = "qwen2.5-1.5b-instruct-q5_k_m.gguf"
model_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", model_name)

llm = Llama(
    model_path,
    n_ctx=4096,
    n_threads=4,
    verbose=False
)

logger.info("RAG successfully loaded")

with open("data/rag_prompt.txt", "r", encoding="utf-8") as f:
    rag_prompt = f.read()

def rag_generation(question: str, retrieved: list[str], instruction: str) -> str:
    
    prompt = f"""{instruction}

Question: {question}

Source:
{retrieved}

Answer:
"""

    print(prompt)

    output = llm(
        prompt,
        max_tokens=100,
        temperature=0.2,
        top_p=0.5
    )
    
    answer = output["choices"][0]["text"].strip()
    last_dot = answer.rfind('.')
    if last_dot != -1:
        answer = answer[:last_dot+1]

    return answer