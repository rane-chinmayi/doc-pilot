import json
import os
import time
from collections import Counter
import numpy as np
import faiss
from google import genai
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
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

# ===== TOOL DEFINITIONS =====
TOOLS = {
    'amplitude': {
        'chunks_file': 'chunks.json',
        'index_file': 'amplitude_index.faiss',
        'embeddings_file': 'embeddings.npy',
        'name': 'Amplitude',
        'docs_url': 'https://amplitude.com/docs'
    },
    'mixpanel': {
        'chunks_file': 'chunks_mixpanel.json',
        'index_file': 'amplitude_index_mixpanel.faiss',
        'embeddings_file': 'embeddings_mixpanel.npy',
        'name': 'Mixpanel',
        'docs_url': 'https://docs.mixpanel.com'
    },
    'google_analytics': {
        'chunks_file': 'chunks_ga.json',
        'index_file': 'amplitude_index_ga.faiss',
        'embeddings_file': 'embeddings_ga.npy',
        'name': 'Google Analytics',
        'docs_url': 'https://support.google.com/analytics'
    }
}

# Load all tools at startup
tool_data = {}
for tool_key, tool_config in TOOLS.items():
    print(f"Loading {tool_config['name']} resources...")
    with open(tool_config['chunks_file'], 'r', encoding='utf-8') as f:
        tool_chunks = json.load(f)
    tool_index = faiss.read_index(tool_config['index_file'])
    tool_data[tool_key] = {'chunks': tool_chunks, 'index': tool_index}
    print(f"  ✓ {len(tool_chunks)} chunks, index loaded")

print("All resources loaded successfully.")

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
    tool_breakdown: list = []
    daily_trend: list = []
    top_questions: list = []
    tool_deflection: list = []
    confidence_breakdown: list = []

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
    # Amplitude
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
    # Mixpanel
    'docs_analysis_reports_funnels.txt': 'https://docs.mixpanel.com/docs/analysis/reports/funnels',
    'docs_analysis_reports_retention.txt': 'https://docs.mixpanel.com/docs/analysis/reports/retention',
    'docs_analysis_reports_insights.txt': 'https://docs.mixpanel.com/docs/analysis/reports/insights',
    'docs_analysis_reports_flows.txt': 'https://docs.mixpanel.com/docs/analysis/reports/flows',
    'docs_features_custom-events.txt': 'https://docs.mixpanel.com/docs/features/custom-events',
    'docs_analysis_users.txt': 'https://docs.mixpanel.com/docs/analysis/users',
    'docs_users_cohorts.txt': 'https://docs.mixpanel.com/docs/users/cohorts',
    # Google Analytics
    'analytics_answer_9304153.txt': 'https://support.google.com/analytics/answer/9304153',
    'analytics_answer_9143382.txt': 'https://support.google.com/analytics/answer/9143382',
    'analytics_answer_9212670.txt': 'https://support.google.com/analytics/answer/9212670',
    'analytics_answer_11986666.txt': 'https://support.google.com/analytics/answer/11986666',
    'analytics_answer_9267568.txt': 'https://support.google.com/analytics/answer/9267568',
    'analytics_answer_9756891.txt': 'https://support.google.com/analytics/answer/9756891',
    'analytics_answer_9356048.txt': 'https://support.google.com/analytics/answer/9356048',
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


def get_answer(query: str, model: str = 'gemini-2.5-flash', tool: str = 'amplitude'):
    """Generate an answer using RAG pipeline for the specified tool"""

    # Get the correct chunks, index, and config for the selected tool
    data = tool_data.get(tool, tool_data['amplitude'])
    chunks = data['chunks']
    index = data['index']
    tool_config = TOOLS.get(tool, TOOLS['amplitude'])

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
    context = ""
    top_source = None
    top_distance = None

    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        context += f"Document {i + 1} (from {chunk['source']}):\n{chunk['text']}\n\n"
        if i == 0:
            top_source = chunk['source']
            top_distance = distances[0][i]

    print(f"Tool: {tool}")
    print(f"Number of chunks available: {len(chunks)}")
    print(f"Top chunk indices: {indices[0]}")
    print(f"Context preview (first 500 chars): {context[:500]}")

    # Build the prompt for Gemini
    prompt = f"""You are a helpful {tool_config['name']} product assistant.
Answer the user's question using ONLY the documentation provided below.
Be concise, practical and specific.
If the answer is genuinely not in the documentation below, say exactly: "I couldn't find this in the {tool_config['name']} documentation. Try searching {tool_config['docs_url']} directly."
Do NOT say you couldn't find it if the answer IS in the documentation.

Documentation:
{context}

Question: {query}

Answer:"""

    print(f"Prompt preview (first 300 chars): {prompt[:300]}")

    # Generate answer with Gemini
    response, model_used = generate_with_retry(prompt, model=model)

    answer = response.text
    source_url = SOURCE_URL_MAP.get(top_source, tool_config['docs_url'])

    return {
        "answer": answer,
        "source": top_source,
        "source_url": source_url,
        "distance": top_distance,
        "model_used": model_used
    }


def get_google_creds(scope):
    """Load Google service account credentials from a local file or environment variable."""
    if os.path.exists('creds.json'):
        # Local development
        return ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
    else:
        # Production — load from environment variable
        creds_json = os.getenv('GOOGLE_CREDS')
        if creds_json:
            creds_dict = json.loads(creds_json)
            return ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        else:
            raise Exception('No Google credentials found')


def log_feedback(query: str, answer: str, feedback: str) -> bool:
    """Log feedback to Google Sheet"""
    try:
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]

        creds = get_google_creds(scope)

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


def detect_tool_from_query(query: str) -> str:
    """Best-effort tool detection from query text when no 'tool' column exists in the sheet."""
    q = (query or '').lower()
    if 'mixpanel' in q:
        return 'Mixpanel'
    if 'google analytics' in q or 'ga4' in q or ' ga ' in q:
        return 'Google Analytics'
    return 'Amplitude'


def detect_confidence_from_record(record: dict) -> str:
    """Best-effort confidence bucket when no 'confidence' column exists in the sheet."""
    confidence = str(record.get('confidence', '')).strip()
    if confidence in ('High', 'Medium', 'Low'):
        return confidence
    # Fall back to feedback as a proxy for confidence
    feedback = record.get('feedback', '')
    if feedback == 'positive':
        return 'High'
    elif feedback == 'negative':
        return 'Low'
    return 'Medium'


EMPTY_ANALYTICS = {
    "total": 0,
    "positive": 0,
    "negative": 0,
    "deflection_rate": 0.0,
    "recent": [],
    "tool_breakdown": [],
    "daily_trend": [],
    "top_questions": [],
    "tool_deflection": [],
    "confidence_breakdown": []
}


def load_analytics_data() -> dict:
    """Load analytics data from Google Sheets"""
    try:
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]

        creds = get_google_creds(scope)

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

        # ----- Tool breakdown & deflection rate per tool -----
        tool_counts = Counter()
        tool_positive_counts = Counter()
        for record in records:
            tool = record.get('tool') or detect_tool_from_query(record.get('query', ''))
            tool_counts[tool] += 1
            if record.get('feedback') == 'positive':
                tool_positive_counts[tool] += 1

        tool_breakdown = [
            {"tool": tool, "queries": count}
            for tool, count in tool_counts.items()
        ]

        tool_deflection = [
            {
                "tool": tool,
                "rate": round((tool_positive_counts[tool] / count * 100), 1) if count > 0 else 0.0
            }
            for tool, count in tool_counts.items()
        ]

        # ----- Daily trend (group by date, first 10 chars of timestamp) -----
        date_counts = Counter()
        for record in records:
            timestamp = record.get('Timestamp', record.get('timestamp', ''))
            date = str(timestamp)[:10] if timestamp else 'unknown'
            date_counts[date] += 1

        daily_trend = [
            {"date": date, "queries": count}
            for date, count in sorted(date_counts.items())
        ]

        # ----- Top questions (most frequently asked) -----
        question_counts = Counter(
            r.get('query', '') for r in records if r.get('query')
        )
        top_questions = [
            {"question": question[:100], "count": count}
            for question, count in question_counts.most_common(5)
        ]

        # ----- Confidence breakdown -----
        confidence_counts = Counter(detect_confidence_from_record(r) for r in records)
        confidence_breakdown = [
            {"level": level, "count": confidence_counts.get(level, 0)}
            for level in ["High", "Medium", "Low"]
        ]

        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "deflection_rate": round(deflection_rate, 2),
            "recent": recent,
            "tool_breakdown": tool_breakdown,
            "daily_trend": daily_trend,
            "top_questions": top_questions,
            "tool_deflection": tool_deflection,
            "confidence_breakdown": confidence_breakdown
        }
    except Exception as e:
        print(f"Error loading analytics data: {e}")
        return dict(EMPTY_ANALYTICS)

# ===== IN-MEMORY CACHE =====
# Cache structure: {cache_key: {data, timestamp}}
answer_cache = {}
related_cache = {}
CACHE_EXPIRY_HOURS = 24


def get_cache_key(query, tool, model):
    return f"{tool}:{model}:{query.lower().strip()}"


def get_from_cache(key):
    if key in answer_cache:
        cached = answer_cache[key]
        # Check if cache is still valid
        if datetime.now() - cached['timestamp'] < timedelta(hours=CACHE_EXPIRY_HOURS):
            print(f"Cache HIT for key: {key[:50]}...")
            return cached['data']
        else:
            # Expired — remove it
            del answer_cache[key]
            print(f"Cache EXPIRED for key: {key[:50]}...")
    return None


def save_to_cache(key, data):
    answer_cache[key] = {
        'data': data,
        'timestamp': datetime.now()
    }
    print(f"Cache SAVED for key: {key[:50]}...")


def get_related_cache_key(query, tool):
    return f"related:{tool}:{query.lower().strip()}"


def get_from_related_cache(key):
    if key in related_cache:
        cached = related_cache[key]
        # Check if cache is still valid
        if datetime.now() - cached['timestamp'] < timedelta(hours=CACHE_EXPIRY_HOURS):
            print(f"Related cache HIT for key: {key[:50]}...")
            return cached['data']
        else:
            # Expired — remove it
            del related_cache[key]
            print(f"Related cache EXPIRED for key: {key[:50]}...")
    return None


def save_to_related_cache(key, data):
    related_cache[key] = {
        'data': data,
        'timestamp': datetime.now()
    }
    print(f"Related cache SAVED for key: {key[:50]}...")


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
    start_time = time.time()

    query = request.get("query", "")
    model = request.get("model", "gemini-2.5-flash")
    tool = request.get("tool", "amplitude")
    print(f"Received request - Query: {query}, Model: {model}, Tool: {tool}")

    tool_config = TOOLS.get(tool, TOOLS['amplitude'])

    try:
        # Check cache first
        cache_key = get_cache_key(query, tool, model)
        cached_result = get_from_cache(cache_key)
        if cached_result:
            print(f"✅ Cache HIT — returning cached answer")
            result_with_flag = dict(cached_result)
            result_with_flag['from_cache'] = True
            result_with_flag['response_time'] = round(time.time() - start_time, 2)
            return result_with_flag

        print(f"❌ Cache MISS — calling Gemini API")

        result = get_answer(query, model=model, tool=tool)
        confidence = get_confidence(result["distance"])

        response_data = {
            "answer": result["answer"],
            "source": result["source"],
            "source_url": result["source_url"],
            "confidence": confidence,
            "distance": float(result["distance"]),
            "model_used": result["model_used"],
            "from_cache": False
        }

        # Save to cache before returning
        save_to_cache(cache_key, response_data)
        print(f"💾 Saved to cache. Total cached: {len(answer_cache)}")

        response_time = round(time.time() - start_time, 2)
        response_data['response_time'] = response_time
        return response_data
    except Exception as e:
        return AskResponse(
            answer=f"Error: {str(e)}",
            source="",
            source_url=tool_config['docs_url'],
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
    - tool_breakdown: Query count per tool
    - daily_trend: Query count per day
    - top_questions: Most frequently asked questions
    - tool_deflection: Deflection rate per tool
    - confidence_breakdown: Count of High/Medium/Low confidence answers
    """
    try:
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        creds = get_google_creds(scope)
        client_gs = gspread.authorize(creds)

        # Debug: list all sheets this account can access
        sheets = client_gs.openall()
        print(f"Available sheets: {[s.title for s in sheets]}")

        sheet = client_gs.open('Amplitude Copilot Logs').sheet1
        rows = sheet.get_all_records()
        print(f"Total rows found: {len(rows)}")
        print(f"First row sample: {rows[0] if rows else 'NO ROWS'}")
    except Exception as e:
        print(f"Debug error: {e}")

    try:
        data = load_analytics_data()
        return AnalyticsResponse(**data)
    except Exception as e:
        return AnalyticsResponse(**EMPTY_ANALYTICS)


@app.post("/related")
async def get_related(request: dict):
    query = request.get("query", "")
    answer = request.get("answer", "")
    confidence = request.get("confidence", "Low")
    model = request.get("model", "gemini-2.5-flash")
    tool = request.get("tool", "amplitude")

    default_questions = [
        "How do I build a funnel in Amplitude?",
        "What is event segmentation?",
        "How do I track retention?"
    ]

    # Skip Gemini call for Low confidence answers
    if confidence == "Low":
        return {"questions": default_questions}

    # Check cache first
    related_cache_key = get_related_cache_key(query, tool)
    cached_questions = get_from_related_cache(related_cache_key)
    if cached_questions:
        return cached_questions

    try:
        prompt = f"""Based on this Amplitude analytics question: "{query}"

Generate exactly 3 short follow-up questions a user might ask next.
Return ONLY a JSON array with exactly 3 strings, nothing else.
Example format: ["Question 1?", "Question 2?", "Question 3?"]

Return only the JSON array, no explanation, no markdown, no backticks."""

        response, _ = generate_with_retry(prompt, model=model)

        text = response.text.strip()
        # Remove markdown backticks if present
        text = text.replace('```json', '').replace('```', '').strip()

        questions = json.loads(text)

        if isinstance(questions, list) and len(questions) >= 3:
            result = {"questions": questions[:3]}
            save_to_related_cache(related_cache_key, result)
            return result
        else:
            return {"questions": default_questions}

    except Exception as e:
        print(f"Related questions error: {e}")
        return {"questions": default_questions}


@app.get("/tools")
async def get_tools():
    """Return the list of available documentation tools."""
    return {"tools": [
        {"key": "amplitude", "name": "Amplitude", "icon": "📊"},
        {"key": "mixpanel", "name": "Mixpanel", "icon": "🔥"},
        {"key": "google_analytics", "name": "Google Analytics", "icon": "📈"}
    ]}


@app.get("/cache/stats")
async def cache_stats():
    return {
        "total_cached": len(answer_cache),
        "keys": [k[:50] for k in answer_cache.keys()],
        "total_related_cached": len(related_cache),
        "related_keys": [k[:50] for k in related_cache.keys()]
    }


@app.delete("/cache/clear")
async def clear_cache():
    answer_cache.clear()
    related_cache.clear()
    return {"status": "cache cleared"}


# ===== RUN SERVER =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
