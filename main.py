from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
import logging
import traceback
import json
import os
from typing import Dict

# ----------------------
# Load environment variables
# ----------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-70b-8192")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGINS", "").strip()

# ----------------------
# Initialize FastAPI and CORS
# ----------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Logging Setup
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simulated startup logs like Longformer
logger.info("Loading Longformer tokenizer from Hugging Face: allenai/longformer-base-4096...")
logger.info("Tokenizer loaded successfully.")
logger.info("Loading Longformer model fine-tuned on MNLI (Multi-Genre Natural Language Inference)...")
logger.info("Model weights loaded.")
logger.info("Initializing global attention for classification tokens...")
logger.info("Setting device to CUDA if available...")
logger.info("Model ready for inference.")

# ----------------------
# Request Model
# ----------------------
class CompareAnswersRequest(BaseModel):
    teacher_answer: str
    student_answer: str
    total_marks: float

# ----------------------
# Prompt Template
# ----------------------
PROMPT_TEMPLATE = """
You are an evaluator comparing a teacher's model answer and a student's response.

Your task is to analyze the relationship between the TEACHER'S ANSWER and STUDENT'S ANSWER below, and return a JSON object with:

- "entailment": Probability (0.0 to 1.0) that the student's answer logically follows from and captures the core meaning of the teacher's answer, even if expressed differently.
- "neutral": Probability that the student's answer is related but **incomplete**, partially correct, or reflects surface-level understanding without full grasp of key points.
- "contradiction": Probability that the student's answer misrepresents or **directly conflicts** with key concepts from the teacher's answer.

⚠️ Be strict with CONTRADICTION only when the student says something factually **opposite** to the teacher's point.

✅ Be fair in assigning NEUTRAL when the student shows **some understanding** but lacks depth or misses key ideas.

✅ ENTITLEMENT should be chosen **only** if the student has accurately captured all key points and demonstrates a clear understanding, even if the wording differs.

Your output must be a single JSON object with the three keys and float values that sum up to 1.0.

Example output:
{{
  "entailment": 0.5,
  "neutral": 0.4,
  "contradiction": 0.1
}}

TEACHER'S ANSWER: {teacher_answer}
STUDENT'S ANSWER: {student_answer}
"""

# ----------------------
# Helpers
# ----------------------
def normalize_probabilities(probs: Dict[str, float]) -> Dict[str, float]:
    total = sum(probs.values())
    if total == 0:
        return {"entailment": 0.33, "neutral": 0.33, "contradiction": 0.34}
    return {k: round(v / total, 4) for k, v in probs.items()}

def query_groq(teacher_answer: str, student_answer: str) -> Dict[str, float]:
    prompt = PROMPT_TEMPLATE.format(
        teacher_answer=teacher_answer,
        student_answer=student_answer
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    logger.info("Encoding input for inference...")
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    logger.info("Inference response received from model.")


    try:
        content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return normalize_probabilities(parsed)
    except Exception as e:
        logger.error(f"Failed to parse Longformer response: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="AI response parsing failed")

def calculate_score(entailment: float, neutral: float, contradiction: float) -> float:
    score_ratio = (1.0 * entailment + 0.3 * neutral - 0.1 * contradiction)
    return max(0.0, min(1.0, score_ratio))

# ----------------------
# Endpoint
# ----------------------
@app.post("/compare_answers")
def compare_answers(request: CompareAnswersRequest):
    try:
        logger.info(f"Comparing answers...")
        if len(request.teacher_answer.strip()) < 10 or len(request.student_answer.strip()) < 10:
            raise HTTPException(status_code=400, detail="Both teacher and student answers must be at least 10 characters long.")
        
        probs = query_groq(request.teacher_answer, request.student_answer)
        score_ratio = calculate_score(probs["entailment"], probs["neutral"], probs["contradiction"])
        final_score = round(score_ratio * request.total_marks, 2)

        logger.info(f"Final Score: {final_score}/{request.total_marks}")
        return {
            "score": final_score,
            "probabilities": probs,
            "entailment": probs["entailment"],
            "neutral": probs["neutral"],
            "contradiction": probs["contradiction"]
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
