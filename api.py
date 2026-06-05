import json
import os
import time
import numpy as np
import faiss
from google import genai
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal

# Load environment variables
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

if not api_key:
    raise ValueError("Error: GOOGLE_API_KEY not found in .env file")

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# Load RAG resources
print("Loading chunks.json...")
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Loading amplitude_index.faiss...")
index = faiss.read_index("amplitude_index.faiss")

print("Resources loaded successfully.")

# ===== REQUEST/RESPONSE MODELS =====
class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    source: str
    source_url: str
    confidence: str
    distance: float
    model_used: str

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    feedback: Literal["positive", "negative"]

class FeedbackResponse(BaseModel):
    status: str

class HealthResponse(BaseModel):
    status: str

class AnalyticsResponse(BaseModel):
    total: int
    positive: int
    negative: int
    deflection_rate: float
    recent: list

class RelatedRequest(BaseModel):
    query: str
    answer: str

class RelatedResponse(BaseModel):
    questions: list

# ===== FASTAPI APP =====
app = FastAPI(
    title="Amplitude AI Assistant API",
    description="RAG-powered API for Amplitude documentation Q&A",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== SOURCE URL MAP =====
SOURCE_URL_MAP = {
    'docs_analytics_charts_funnel-analysis.txt': 'https://amplitude.com/docs/analytics/charts/funnel-analysis',
    'docs_analytics_charts_event-segmentation.txt': 'https://amplitude.com/docs/analytics/charts/event-segmentation',
    'docs_analytics_charts_retention-analysis.txt': 'https://amplitude.com/docs/analytics/charts/retention-analysis',
    'docs_analytics_behavioral-cohorts.txt': 'https://amplitude.com/docs/analytics/behavioral-cohorts',
    'docs_analytics_charts_data-tables.txt': 'https://amplitude.com/docs/analytics/charts/data-tables',
    'docs_analytics_user-data-lookup.txt': 'https://amplitude.com/docs/analytics/user-data-lookup',
    'docs_get-started_understand-user-activity.txt': 'https://amplitude.com/docs/get-started/understand-user-activity',
    'docs_analytics_charts_compass.txt': 'https://amplitude.com/docs/analytics/charts/compass',
    'docs_analytics_charts_lifecycle.txt': 'https://amplitude.com/docs/analytics/charts/lifecycle',
    'docs_analytics_charts_stickiness.txt': 'https://amplitude.com/docs/analytics/charts/stickiness',
    'docs_analytics_charts_revenue-ltv.txt': 'https://amplitude.com/docs/analytics/charts/revenue-ltv',
    'docs_analytics_account-level-reporting.txt': 'https://amplitude.com/docs/analytics/account-level-reporting',
    'docs_analytics_charts_build-charts-add-user-segments.txt': 'https://amplitude.com/docs/analytics/charts/build-charts-add-user-segments',
    'docs_analytics_charts_journeys_journeys-understand-paths.txt': 'https://amplitude.com/docs/analytics/charts/journeys/journeys-understand-paths',
    'docs_analytics_charts_journeys_journeys-understand-visualizations.txt': 'https://amplitude.com/docs/analytics/charts/journeys/journeys-understand-visualizations',
    'docs_analytics_charts_event-segmentation_event-segmentation-build.txt': 'https://amplitude.com/docs/analytics/charts/event-segmentation/event-segmentation-build',
}

# ===== RETRY MECHANISM =====
MODEL_FALLBACK_CHAIN = [
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
    'gemini-2.0-flash-lite',
    'gemini-1.5-flash-latest'
]

def generate_with_retry(prompt, model='gemini-2.5-flash', retries=2, delay=3):
    # First try the requested model
    models_to_try = [model]

    # Add fallback models that aren't the same as requested
    for m in MODEL_FALLBACK_CHAIN:
        if m != model and m not in models_to_try:
            models_to_try.append(m)

    last_error = None
    for current_model in models_to_try:
        for attempt in range(retries):
            try:
                print(f"Trying model: {current_model}")
                response = client.models.generate_content(
                    model=current_model,
                    contents=prompt
                )
                return response, current_model
            except Exception as e:
                last_error = e
                if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                    print(f"Quota exceeded for {current_model}, trying next model...")
                    break  # Try next model
                elif '503' in str(e) or 'UNAVAILABLE' in str(e):
                    if attempt < retries - 1:
                        time.sleep(delay)
                    else:
                        break
                else:
                    raise e

    raise last_error

# ===== HELPER FUNCTIONS =====
def get_confidence(distance):
    """Get confidence level based on FAISS distance score"""
    if distance < 0.45:
        return "High"
    elif distance < 0.65:
        return "Medium"
    else:
        return "Low"


def get_answer(query: str, model: str = 'gemini-2.5-flash'):
    """Generate an answer using RAG pipeline"""

    # Embed the query
    response = client.models.embed_content(
        model='gemini-embedding-001',
        contents=query
    )
    embedding = response.embeddings[0].values
    query_embedding = np.array(embedding).astype("float32").reshape(1, -1)

    # Search FAISS index for top 3 chunks
    distances, indices = index.search(query_embedding, 3)

    # Build context from retrieved chunks
    context = "Amplitude Documentation:\n\n"
    top_source = None
    top_distance = None

    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        context += f"Document {i + 1} (from {chunk['source']}):\n{chunk['text']}\n\n"
        if i == 0:
            top_source = chunk['source']
            top_distance = distances[0][i]

    # Build the prompt for Gemini
    system_prompt = """You are an expert Amplitude analytics assistant. Answer questions based ONLY on the provided Amplitude documentation.

Instructions:
- Answer only using information from the provided documentation
- Keep answers concise and practical
- If the answer isn't in the provided documentation, respond with: "I couldn't find this in the Amplitude documentation. Try searching docs.amplitude.com directly."
- Provide actionable and clear answers"""

    user_prompt = f"""{system_prompt}

{context}

User Question: {query}

Please answer based on the documentation above."""

    # Generate answer with Gemini
    response, model_used = generate_with_retry(user_prompt, model=model)

    answer = response.text
    source_url = SOURCE_URL_MAP.get(top_source, 'https://amplitude.com/docs')

    return {
        "answer": answer,
        "source": top_source,
        "source_url": source_url,
        "distance": top_distance,
        "model_used": model_used
    }


def log_feedback(query: str, answer: str, feedback: str) -> bool:
    """Log feedback to Google Sheet"""
    try:
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]

        # Local development: load from creds.json
        if os.path.exists('creds.json'):
            creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
        # Production: would need to load from environment
        else:
            print("Warning: creds.json not found. Feedback not logged.")
            return False

        client_gs = gspread.authorize(creds)

        # Open the sheet
        sheet = client_gs.open("Amplitude Copilot Logs").sheet1

        # Append row
        timestamp = datetime.now().isoformat()
        answer_preview = answer[:500]
        sheet.append_row([timestamp, query, answer_preview, feedback])

        return True
    except Exception as e:
        print(f"Error logging feedback: {e}")
        return False


def load_analytics_data() -> dict:
    """Load analytics data from Google Sheets"""
    try:
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]

        # Local development: load from creds.json
        if os.path.exists('creds.json'):
            creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
        # Production: would need to load from environment
        else:
            print("Warning: creds.json not found. Analytics not available.")
            return {
                "total": 0,
                "positive": 0,
                "negative": 0,
                "deflection_rate": 0.0,
                "recent": []
            }

        client_gs = gspread.authorize(creds)

        # Open the sheet
        sheet = client_gs.open("Amplitude Copilot Logs").sheet1
        records = sheet.get_all_records()

        # Calculate metrics
        total = len(records)
        positive = sum(1 for r in records if r.get('feedback') == 'positive')
        negative = sum(1 for r in records if r.get('feedback') == 'negative')
        deflection_rate = (positive / total * 100) if total > 0 else 0.0

        # Get last 10 queries
        recent = []
        for record in records[-10:]:
            recent.append({
                "timestamp": record.get('Timestamp', record.get('timestamp', '')),
                "query": record.get('query', '')[:100],  # Truncate to 100 chars
                "feedback": record.get('feedback', '')
            })

        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "deflection_rate": round(deflection_rate, 2),
            "recent": recent
        }
    except Exception as e:
        print(f"Error loading analytics data: {e}")
        return {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "deflection_rate": 0.0,
            "recent": []
        }

# ===== IN-MEMORY CACHE =====
answer_cache = {}

# ===== API ENDPOINTS =====

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="ok")


@app.post("/ask")
async def ask(request: dict):
    """
    Ask a question about Amplitude documentation.

    Returns:
    - answer: The generated answer
    - source: The source document
    - source_url: Link to the source documentation page
    - confidence: Confidence level (High/Medium/Low)
    - distance: FAISS similarity distance score
    """
    query = request.get("query", "")
    model = request.get("model", "gemini-2.5-flash")
    print(f"Received request - Query: {query}, Model: {model}")

    try:
        # Cache key includes model so different models don't share results
        cache_key = f"{query}::{model}"
        if cache_key in answer_cache:
            return answer_cache[cache_key]

        result = get_answer(query, model=model)
        confidence = get_confidence(result["distance"])

        response = AskResponse(
            answer=result["answer"],
            source=result["source"],
            source_url=result["source_url"],
            confidence=confidence,
            distance=float(result["distance"]),
            model_used=result["model_used"]
        )

        # Cache the result before returning
        answer_cache[cache_key] = response
        return response
    except Exception as e:
        return AskResponse(
            answer=f"Error: {str(e)}",
            source="",
            source_url="https://amplitude.com/docs",
            confidence="Low",
            distance=1.0,
            model_used=model
        )


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Log user feedback about an answer.

    Feedback is logged to Google Sheets if creds.json is available.
    """
    try:
        success = log_feedback(request.query, request.answer, request.feedback)

        if success:
            return FeedbackResponse(status="logged")
        else:
            return FeedbackResponse(status="not_logged_no_credentials")
    except Exception as e:
        return FeedbackResponse(status=f"error: {str(e)}")


@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """
    Get analytics data from Google Sheets.

    Returns:
    - total: Total number of queries
    - positive: Count of positive feedback
    - negative: Count of negative feedback
    - deflection_rate: Percentage of positive feedback
    - recent: Last 10 queries with timestamp and feedback
    """
    try:
        data = load_analytics_data()
        return AnalyticsResponse(**data)
    except Exception as e:
        return AnalyticsResponse(
            total=0,
            positive=0,
            negative=0,
            deflection_rate=0.0,
            recent=[]
        )


@app.post("/related")
async def get_related(request: dict):
    query = request.get("query", "")
    answer = request.get("answer", "")
    confidence = request.get("confidence", "Low")
    model = request.get("model", "gemini-2.5-flash")

    default_questions = [
        "How do I build a funnel in Amplitude?",
        "What is event segmentation?",
        "How do I track retention?"
    ]

    # Skip Gemini call for Low confidence answers
    if confidence == "Low":
        return {"questions": default_questions}

    try:
        prompt = f"""Based on this Amplitude analytics question: "{query}"

Generate exactly 3 short follow-up questions a user might ask next.
Return ONLY a JSON array with exactly 3 strings, nothing else.
Example format: ["Question 1?", "Question 2?", "Question 3?"]

Return only the JSON array, no explanation, no markdown, no backticks."""

        response = generate_with_retry(prompt, model=model)

        text = response.text.strip()
        # Remove markdown backticks if present
        text = text.replace('```json', '').replace('```', '').strip()

        questions = json.loads(text)

        if isinstance(questions, list) and len(questions) >= 3:
            return {"questions": questions[:3]}
        else:
            return {"questions": default_questions}

    except Exception as e:
        print(f"Related questions error: {e}")
        return {"questions": default_questions}


# ===== RUN SERVER =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
