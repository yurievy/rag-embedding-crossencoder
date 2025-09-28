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
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# -------------------------------
# 4. Local modules
# -------------------------------
from pipeline import process_question

# -------------------------------
# 5. Initialize FastAPI app
# -------------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Подключение папки static (для CSS, JS, шрифтов, картинок и т.п.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------
# 6. Pydantic models
# -------------------------------
class QuestionInput(BaseModel):
    question: str
    use_RAG: bool

# -------------------------------
# 7. Routes
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> Any:
    """
    Render the main HTML page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask")
async def ask(data: QuestionInput) -> Any:
    """
    Handle a user question:
        - Parse JSON payload via Pydantic model
        - Process question using pipeline
        - Return formatted JSON response
    """
    result = process_question(data.dict())
    return JSONResponse({"answer": result})


# -------------------------------
# 8. Run server (only via uvicorn)
# -------------------------------
# # uvicorn server_fastapi:app --reload
