import os
import logging
from llama_cpp import Llama
from typing import List

logger = logging.getLogger(__name__)

logger.info("Initializing RAG...")

# Model path
MODEL_NAME = "qwen2.5-1.5b-instruct-q5_k_m.gguf"
MODEL_PATH = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", MODEL_NAME)

# Initialize Llama model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=4,
    verbose=False
)

logger.info("RAG successfully loaded")

# Load prompt template
PROMPT_PATH = "data/rag_prompt.txt"
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    rag_prompt = f.read()


def rag_generation(question: str, retrieved: List[str], instruction: str) -> str:
    """
    Generate an answer based on a question and retrieved sources.

    Args:
        question (str): The user's question.
        retrieved (List[str]): List of retrieved documents/sources.
        instruction (str): Instruction to guide the model.

    Returns:
        str: Generated answer.
    """
    # Build the prompt
    prompt = f"""{instruction}

Question: {question}

Source:
{retrieved}

Answer:
"""

    # Generate output from the model
    output = llm(
        prompt,
        max_tokens=100,
        temperature=0.2,
        top_p=0.5
    )

    answer = output["choices"][0]["text"].strip()

    # Trim the answer to the last period
    last_dot = answer.rfind(".")
    if last_dot != -1:
        answer = answer[:last_dot + 1]

    return answer
