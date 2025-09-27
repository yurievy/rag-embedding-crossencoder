import os
import logging
from typing import List
from llama_cpp import Llama

# -------------------------------
# Configure logger
# -------------------------------
logger = logging.getLogger(__name__)
logger.info("Initializing RAG...")

# -------------------------------
# Constants
# -------------------------------
MODEL_NAME = "qwen2.5-1.5b-instruct-q5_k_m.gguf"
MODEL_PATH = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", MODEL_NAME)
PROMPT_PATH = "data/rag_prompt.txt"

# -------------------------------
# Initialize Llama model
# -------------------------------
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=4,
    verbose=False
)
logger.info(f"RAG model '{MODEL_NAME}' successfully loaded.")

# -------------------------------
# Load RAG prompt template
# -------------------------------
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    rag_prompt = f.read()


# -------------------------------
# Function to generate RAG answer
# -------------------------------
def rag_generation(question: str, retrieved: List[str], instruction: str) -> str:
    """
    Generate an answer based on a question and retrieved sources using RAG.

    Args:
        question (str): The user's question.
        retrieved (List[str]): List of retrieved documents/sources.
        instruction (str): Instruction to guide the model.

    Returns:
        str: Generated answer.
    """
    # Build the prompt for the model
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

    # Trim answer to the last period to avoid incomplete sentences
    last_dot = answer.rfind(".")
    if last_dot != -1:
        answer = answer[:last_dot + 1]

    return answer
