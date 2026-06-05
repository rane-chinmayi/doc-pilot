import streamlit as st
import json
import os
import numpy as np
import faiss
from google import genai
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pandas as pd
import re
from collections import Counter

# Load environment variables
load_dotenv()
api_key = st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else os.getenv('GOOGLE_API_KEY')

if not api_key:
    st.error("Error: GEMINI_API_KEY not found in Streamlit secrets or .env file")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Amplitude AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ===== GLOBAL STYLES =====
st.markdown("""
<style>
    /* Dark background */
    .stApp {
        background-color: #0F1117;
    }

    /* Hide default Streamlit header */
    header[data-testid="stHeader"] {
        background-color: #0F1117;
    }

    /* Main container width */
    .block-container {
        max-width: 800px;
        padding-top: 3rem;
    }

    /* Remove default spacing */
    * {
        color: #F0F0F0;
    }

    /* Example chips styling */
    div[data-testid="stButton"] button {
        background: transparent !important;
        border: 1px solid #2E3250 !important;
        color: #8B8FA8 !important;
        border-radius: 20px !important;
        font-size: 13px !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stButton"] button:hover {
        border-color: #7C3AED !important;
        color: #7C3AED !important;
    }

    /* Search input styling */
    div[data-testid="stTextInput"] input {
        background-color: #1A1D2E !important;
        border: 1px solid #2E3250 !important;
        border-radius: 12px !important;
        color: #F0F0F0 !important;
        font-size: 15px !important;
        padding: 12px 20px !important;
        height: 52px !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #6B7280 !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #7C3AED !important;
        box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2) !important;
        outline: none !important;
    }

    /* Answer card styling */
    .answer-card {
        background: #1A1D2E;
        border-left: 3px solid #7C3AED;
        border-radius: 8px;
        padding: 20px 24px;
        margin-top: 16px;
    }

    /* Confidence badge */
    .confidence-badge {
        font-size: 13px;
        margin-bottom: 12px;
        font-weight: 500;
    }

    /* Helper text */
    .helper-text {
        font-size: 12px;
        color: #6B7280;
        margin-top: 8px;
        text-align: center;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6B7280;
        font-size: 12px;
        margin-top: 60px;
        padding-top: 40px;
        border-top: 1px solid #2E3250;
    }
</style>
""", unsafe_allow_html=True)

# Load resources with caching
@st.cache_resource
def load_resources():
    """Load chunks, index, and initialize client"""
    try:
        with open("chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)

        index = faiss.read_index("amplitude_index.faiss")
        client = genai.Client(api_key=api_key)

        return chunks, index, client
    except FileNotFoundError as e:
        st.error(f"Error: Required file not found - {e}")
        st.stop()

chunks, index, client = load_resources()


def get_confidence(distance):
    """Get confidence level based on FAISS distance score"""
    if distance < 0.45:
        return "High", "🟢"
    elif distance < 0.65:
        return "Medium", "🟡"
    else:
        return "Low", "🔴"


def get_answer(query):
    """Generate an answer using RAG pipeline"""

    # Embed the query
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )
    query_embedding = np.array(response.embeddings[0].values).astype("float32").reshape(1, -1)

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
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=user_prompt
    )

    answer = response.text

    return {
        "answer": answer,
        "source": top_source,
        "distance": top_distance
    }


def log_feedback(query, answer, feedback):
    """Log feedback to Google Sheet"""
    try:
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]

        # Local development: load from creds.json
        if os.path.exists('creds.json'):
            creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
        # Streamlit Cloud: load from secrets
        else:
            creds_dict = json.loads(st.secrets["GOOGLE_CREDS"])
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)

        client_gs = gspread.authorize(creds)

        # Open the sheet
        sheet = client_gs.open("Amplitude Copilot Logs").sheet1

        # Append row
        timestamp = datetime.now().isoformat()
        answer_preview = answer[:500]
        sheet.append_row([timestamp, query, answer_preview, feedback])

        return True
    except Exception as e:
        return False


def load_analytics_data():
    """Load data from Google Sheet"""
    try:
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]

        # Local development: load from creds.json
        if os.path.exists('creds.json'):
            creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
        # Streamlit Cloud: load from secrets
        else:
            creds_dict = json.loads(st.secrets["GOOGLE_CREDS"])
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)

        client_gs = gspread.authorize(creds)

        sheet = client_gs.open("Amplitude Copilot Logs").sheet1
        records = sheet.get_all_records()

        return pd.DataFrame(records)
    except Exception as e:
        return None


# ===== CREATE TABS =====
tab1, tab2 = st.tabs(["🤖 Assistant", "📊 Analytics"])

# ===== TAB 1: ASSISTANT =====
with tab1:
    # Initialize session state
    if 'last_query' not in st.session_state:
        st.session_state.last_query = None
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = False

    # HEADER
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🤖</div>
        <h1 style="font-size: 2.2rem; font-weight: 700; margin: 0; color: #F0F0F0;">Amplitude AI Assistant</h1>
        <p style="font-size: 1rem; color: #8B8FA8; margin-top: 0.5rem;">Get instant answers from Amplitude documentation</p>
    </div>
    """, unsafe_allow_html=True)

    # EXAMPLE CHIPS
    col1, col2, col3 = st.columns(3, gap="small")
    example_questions = [
        "How do I build a funnel?",
        "What is a cohort?",
        "How do I track retention?"
    ]

    with col1:
        if st.button(example_questions[0], key="chip1", use_container_width=True):
            st.session_state.last_query = example_questions[0]
            st.session_state.last_result = None
            st.session_state.feedback_given = False
            st.rerun()

    with col2:
        if st.button(example_questions[1], key="chip2", use_container_width=True):
            st.session_state.last_query = example_questions[1]
            st.session_state.last_result = None
            st.session_state.feedback_given = False
            st.rerun()

    with col3:
        if st.button(example_questions[2], key="chip3", use_container_width=True):
            st.session_state.last_query = example_questions[2]
            st.session_state.last_result = None
            st.session_state.feedback_given = False
            st.rerun()

    # SEARCH BAR
    st.markdown('<div style="margin-top: 2rem; margin-bottom: 0.5rem;"></div>', unsafe_allow_html=True)
    query = st.text_input(
        "Ask a question",
        placeholder="Ask me anything about Amplitude...",
        label_visibility="collapsed"
    )

    st.markdown('<div class="helper-text">Press Enter to search</div>', unsafe_allow_html=True)

    # QUERY SUBMISSION - Store in session state
    if query:
        try:
            with st.status("Processing your question...", expanded=True) as status:
                status.update(label="🔍 Searching documentation...", state="running")

                # Get answer and store in session state
                result = get_answer(query)
                st.session_state.last_query = query
                st.session_state.last_result = result
                st.session_state.feedback_given = False

                status.update(label="✓ Done", state="complete")

        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "ServerError" in error_str or "high demand" in error_str:
                st.warning("⚠️ Gemini is experiencing high demand. Please try again in a moment.")
            else:
                st.error(f"An error occurred: {e}")

    # ANSWER CARD & FEEDBACK - Display from session state (persists across reruns)
    if st.session_state.last_result is not None:
        confidence_level, confidence_emoji = get_confidence(st.session_state.last_result["distance"])

        st.markdown(f"""
        <div class="answer-card">
            <div class="confidence-badge">{confidence_emoji} {confidence_level} Confidence</div>
            <div style="color: #F0F0F0; line-height: 1.6;">{st.session_state.last_result["answer"]}</div>
            <div style="color: #7C3AED; font-size: 13px; margin-top: 16px;">📄 {st.session_state.last_result["source"]}</div>
        </div>
        """, unsafe_allow_html=True)

        # Feedback buttons
        if not st.session_state.feedback_given:
            st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 4])

            with col1:
                if st.button("👍 Helpful", key="feedback_yes", use_container_width=True):
                    log_feedback(st.session_state.last_query, st.session_state.last_result["answer"], "positive")
                    st.session_state.feedback_given = True
                    st.success("Thank you!")
                    st.rerun()

            with col2:
                if st.button("👎 Not Helpful", key="feedback_no", use_container_width=True):
                    log_feedback(st.session_state.last_query, st.session_state.last_result["answer"], "negative")
                    st.session_state.feedback_given = True
                    st.info("Got it, thanks!")
                    st.rerun()
        else:
            st.markdown('<div style="margin-top: 2rem; text-align: center; color: #6B7280; font-size: 13px;">Feedback submitted ✓</div>', unsafe_allow_html=True)

    # FOOTER
    st.markdown("""
    <div class="footer">
        Built with Amplitude Docs · Gemini AI · FAISS
    </div>
    """, unsafe_allow_html=True)

# ===== TAB 2: ANALYTICS =====
with tab2:
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True, key="refresh_analytics"):
        st.rerun()

    # Load data
    df = load_analytics_data()

    if df is None or df.empty:
        st.warning("No data available yet. Start asking questions in the Assistant tab to generate analytics.")
    else:
        # Calculate metrics
        total_queries = len(df)
        positive_feedback = len(df[df.get('feedback', '') == 'positive'])
        negative_feedback = len(df[df.get('feedback', '') == 'negative'])
        deflection_rate = (positive_feedback / total_queries * 100) if total_queries > 0 else 0

        # Display metrics
        st.markdown("### Key Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Queries", total_queries)

        with col2:
            st.metric("Positive Feedback", positive_feedback)

        with col3:
            st.metric("Negative Feedback", negative_feedback)

        with col4:
            st.metric("Deflection Rate", f"{deflection_rate:.1f}%")

        # Query topics analysis
        st.markdown("### Query Topics")

        if 'query' in df.columns:
            keywords = []
            for query_text in df['query'].dropna():
                words = re.findall(r'\b[a-z]+\b', str(query_text).lower())
                stop_words = {'how', 'do', 'i', 'the', 'a', 'to', 'can', 'what', 'is', 'in', 'and', 'or', 'if', 'for', 'on', 'at', 'by'}
                words = [w for w in words if w not in stop_words and len(w) > 2]
                keywords.extend(words)

            keyword_counts = Counter(keywords)
            top_keywords = dict(keyword_counts.most_common(8))

            if top_keywords:
                chart_df = pd.DataFrame(list(top_keywords.items()), columns=['Topic', 'Count'])
                st.bar_chart(chart_df.set_index('Topic'))
            else:
                st.info("Not enough data to generate topic chart yet.")

        # Recent queries table
        st.markdown("### Recent Queries (Last 10)")

        if 'query' in df.columns and 'feedback' in df.columns:
            recent_df = df[['Timestamp' if 'Timestamp' in df.columns else 'timestamp', 'query', 'feedback']].tail(10).copy()
            st.dataframe(recent_df, use_container_width=True, hide_index=True)
        else:
            st.info("No query data available yet.")
