# -------------------------------
# 1. Configure logging first
# -------------------------------
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Server starting...")

# -------------------------------
# 2. Standard libraries
# -------------------------------
from typing import Any

# -------------------------------
# 3. Third-party libraries
# -------------------------------
from flask import Flask, request, jsonify, render_template

# -------------------------------
# 4. Local modules
# -------------------------------
from pipeline import process_question

# -------------------------------
# 5. Initialize Flask app
# -------------------------------
app = Flask(__name__)

# -------------------------------
# 7. Routes
# -------------------------------
@app.route("/")
def home() -> str:
    """
    Render the main HTML page.
    """
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask() -> Any:
    """
    Handle a user question:
        - Extract JSON payload
        - Process question using pipeline
        - Return formatted HTML response
    """
    data = request.json or {}
    
    result = process_question(data)

    return jsonify({"answer": result})


# -------------------------------
# 8. Run server
# -------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
