# DocPilot — AI Analytics Assistant

## Overview

DocPilot is a RAG-based AI copilot that answers questions about analytics tools (Amplitude, Mixpanel, Google Analytics) using official documentation as the knowledge base. Built as a portfolio project to demonstrate AI PM skills.

## Live Demo

- Frontend: [https://amplitude-copilot-frontend.vercel.app](https://docpilot-analytics.vercel.app/)
- Backend API: [https://docpilot-api-v061.onrender.com](https://docpilot-api-v061.onrender.com/)

## Architecture

- User asks a question in the React frontend
- Frontend sends a POST request to the FastAPI backend
- Backend embeds the query using the Gemini embedding API
- FAISS index is searched for the top 3 relevant chunks
- Retrieved chunks + query are sent to Gemini for answer generation
- Answer is returned to the frontend with a confidence score and source URL
- Feedback is logged to Google Sheets

```
User Query
    ↓
React Frontend (Vercel)
    ↓ POST /ask
FastAPI Backend (Render)
    ↓
Gemini Embedding API
    ↓
FAISS Vector Search
    ↓
Top 3 Relevant Chunks
    ↓
Gemini Generation API
    ↓
Answer + Confidence + Source
    ↓
React Frontend
    ↓
Google Sheets (Feedback Logging)
```

## Tech Stack

- **React + Vite** — Frontend framework
- **Tailwind CSS** — Styling
- **FastAPI** — Python backend API
- **FAISS** — Vector similarity search
- **Gemini API** — Embeddings and answer generation
- **Google Sheets** — Query and feedback logging
- **Vercel** — Frontend hosting
- **Render** — Backend hosting

## Knowledge Base

- Amplitude: 37 chunks from 16 documentation pages
- Mixpanel: 39 chunks from 7 documentation pages
- Google Analytics: 69 chunks from 7 documentation pages
- Total: 145 chunks

## Features

- Multi-turn chat interface
- Multi-tool support (Amplitude, Mixpanel, Google Analytics)
- Confidence scoring (High/Medium/Low)
- Related questions suggestions
- Copy answer button
- Source documentation links
- Search history (persisted in localStorage, tool-specific)
- Model selector with automatic fallback chain
- Dark/Light mode
- Analytics dashboard with 5 charts
- In-memory answer caching
- Keyboard shortcuts (Ctrl+K)
- FAQ chips from real analytics data

## API Endpoints

- `POST /ask` — Get answer for a query
- `POST /feedback` — Log user feedback
- `POST /related` — Get related questions
- `GET /analytics` — Get usage metrics
- `GET /tools` — Get available tools
- `GET /health` — Health check
- `GET /cache/stats` — Cache statistics
- `DELETE /cache/clear` — Clear cache

## Local Development Setup

1. Clone both repos (`amplitude-copilot` for the backend, `amplitude-copilot-frontend` for the frontend)
2. Install Python dependencies: `pip install -r requirements.txt`
3. Set up a `.env` file in `amplitude-copilot` with `GOOGLE_API_KEY`
4. Run `python api.py` for the backend
5. `cd amplitude-copilot-frontend && npm install && npm run dev`
6. Open http://localhost:5173

## Environment Variables

- `GOOGLE_API_KEY` — Gemini API key from aistudio.google.com
- `GOOGLE_CREDS` — Google service account credentials JSON

## Project Structure

```
amplitude-copilot/                  # Backend (FastAPI)
├── api.py                          # Main API: /ask, /feedback, /related, /analytics, /tools, /health, /cache/*
├── scrape_docs.py                  # Scrapes Amplitude documentation
├── scrape_mixpanel.py              # Scrapes Mixpanel documentation
├── scrape_ga.py                    # Scrapes Google Analytics documentation
├── chunk_docs.py                   # Chunks Amplitude docs into chunks.json
├── chunk_mixpanel.py               # Chunks Mixpanel docs into chunks_mixpanel.json
├── chunk_ga.py                     # Chunks GA docs into chunks_ga.json
├── build_index.py                  # Builds FAISS index + embeddings for Amplitude
├── build_index_mixpanel.py         # Builds FAISS index + embeddings for Mixpanel
├── build_index_ga.py               # Builds FAISS index + embeddings for Google Analytics
├── chunks.json / chunks_mixpanel.json / chunks_ga.json
├── amplitude_index.faiss / amplitude_index_mixpanel.faiss / amplitude_index_ga.faiss
├── embeddings.npy / embeddings_mixpanel.npy / embeddings_ga.npy
├── docs/ / docs_mixpanel/ / docs_ga/   # Scraped raw documentation (.txt)
├── requirements.txt
├── creds.json                      # Google service account credentials (local dev, gitignored)
└── .env                             # GOOGLE_API_KEY (gitignored)

amplitude-copilot-frontend/         # Frontend (React + Vite)
├── src/
│   ├── App.jsx                     # Main UI: chat, tool selector, analytics dashboard
│   ├── App.css
│   ├── index.css
│   ├── main.jsx
│   └── assets/
├── index.html
├── package.json
├── vite.config.js
└── tailwind.config.js
```

## Metrics

- Total Queries: [update with real number]
- Deflection Rate: [update with real number]
- Positive Feedback: [update with real number]

## Built By

Developed by Chinmayi as a portfolio project demonstrating AI PM skills.

LinkedIn: https://www.linkedin.com/in/chinmayi-rane/
