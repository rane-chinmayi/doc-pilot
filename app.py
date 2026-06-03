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

# Load environment variables
load_dotenv()
api_key = st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else os.getenv('GOOGLE_API_KEY')

if not api_key:
    st.error("Error: GEMINI_API_KEY not found in Streamlit secrets or .env file")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Amplitude Analytics Copilot",
    page_icon="📊",
    layout="wide"
)

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

    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        context += f"Document {i + 1} (from {chunk['source']}):\n{chunk['text']}\n\n"
        if i == 0:
            top_source = chunk['source']

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
        model="gemini-2.5-flash",
        contents=user_prompt
    )

    answer = response.text

    return {
        "answer": answer,
        "source": top_source
    }


def log_feedback(query, answer, feedback):
    """Log feedback to Google Sheet"""
    try:
        # Load credentials from Streamlit secrets
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
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
    except KeyError:
        st.warning("Note: GOOGLE_CREDS not found in Streamlit secrets. Feedback not logged to Google Sheets.")
        return False
    except Exception as e:
        st.warning(f"Note: Could not log feedback - {e}")
        return False


# UI
st.title("📊 Amplitude Analytics Copilot")
st.caption("Ask questions about Amplitude analytics features and get instant answers from our documentation.")

# Query input
query = st.text_input(
    "What do you need help with?",
    placeholder="e.g. How do I build a funnel in Amplitude?"
)

# Process query
if query:
    try:
        with st.status("Processing your query...", expanded=True) as status:
            status.update(label="🔍 Searching Amplitude docs...", state="running")

            # Embed query and search
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=query
            )
            query_embedding = np.array(response.embeddings[0].values).astype("float32").reshape(1, -1)
            distances, indices = index.search(query_embedding, 3)

            # Build context
            context = "Amplitude Documentation:\n\n"
            top_source = None
            for i, idx in enumerate(indices[0]):
                chunk = chunks[idx]
                context += f"Document {i + 1} (from {chunk['source']}):\n{chunk['text']}\n\n"
                if i == 0:
                    top_source = chunk['source']

            status.update(label="🤖 Generating answer...", state="running")

            # Generate answer
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

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt
            )
            answer = response.text

            status.update(label="✅ Answer ready!", state="complete")

            result = {
                "answer": answer,
                "source": top_source
            }

        # Display answer
        st.subheader("Answer")
        st.write(result["answer"])

        # Display source
        st.caption(f"📄 Source: {result['source']}")

        # Feedback section
        st.divider()
        st.subheader("Was this helpful?")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Yes"):
                if log_feedback(query, result["answer"], "positive"):
                    st.success("Thank you for your feedback!")
                else:
                    st.success("Thank you for your feedback!")

        with col2:
            if st.button("👎 No"):
                if log_feedback(query, result["answer"], "negative"):
                    st.info("We appreciate your feedback and will use it to improve.")
                else:
                    st.info("We appreciate your feedback and will use it to improve.")

    except Exception as e:
        error_str = str(e)
        if "503" in error_str or "ServerError" in error_str or "high demand" in error_str:
            st.warning("Gemini is experiencing high demand right now. Please wait a moment and try again.")
        else:
            st.error(f"An error occurred: {e}")
