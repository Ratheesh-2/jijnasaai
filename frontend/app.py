import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env so the Streamlit process sees the same env vars as the backend
load_dotenv(Path(__file__).parent.parent / ".env")

import streamlit as st
from frontend.api_client import APIClient
from frontend.components.sidebar import render_sidebar
from frontend.components.chat_view import render_chat
from frontend.components.document_upload import render_document_upload
from frontend.components.voice_controls import render_voice_controls
from frontend.components.admin_dashboard import render_admin_dashboard

st.set_page_config(
    page_title="JijnasaAI",
    page_icon="https://em-content.zobj.net/source/twitter/408/high-voltage_26a1.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Gradient header */
    .stApp > header { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown { color: #e0e0e0; }

    /* Chat message styling */
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }

    /* Provider badges */
    .provider-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7em;
        font-weight: 600;
        margin-left: 4px;
    }
    .provider-openai { background: #10a37f20; color: #10a37f; border: 1px solid #10a37f40; }
    .provider-anthropic { background: #d4a27420; color: #d4a274; border: 1px solid #d4a27440; }
    .provider-google { background: #4285f420; color: #4285f4; border: 1px solid #4285f440; }
    .provider-perplexity { background: #1fb8cd20; color: #1fb8cd; border: 1px solid #1fb8cd40; }

    /* Suggested prompt buttons */
    .stButton > button[kind="secondary"] {
        border-radius: 20px;
        border: 1px solid #667eea40;
        transition: all 0.2s;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #667eea;
        background: #667eea10;
    }

    /* Cost progress bar */
    .stProgress > div > div { background: linear-gradient(90deg, #10a37f, #667eea); }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Landing page styles */
    .landing-hero {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .landing-logo {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 800;
        margin-bottom: 0.2em;
        letter-spacing: -0.02em;
    }
    .landing-tagline {
        color: #999;
        font-size: 0.95em;
        margin-bottom: 1.5em;
        font-style: italic;
    }
    .landing-headline {
        font-size: 1.6em;
        font-weight: 700;
        color: #e0e0e0;
        line-height: 1.3;
        margin-bottom: 0.3em;
    }
    .landing-subheadline {
        font-size: 1.05em;
        color: #aaa;
        margin-bottom: 2em;
    }
    .landing-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        min-height: 180px;
    }
    .landing-card-icon {
        font-size: 2em;
        margin-bottom: 0.5em;
    }
    .landing-card h3 {
        color: #e0e0e0;
        font-size: 1.05em;
        margin-bottom: 0.5em;
    }
    .landing-card p {
        color: #999;
        font-size: 0.9em;
        line-height: 1.5;
    }
    .landing-audience {
        text-align: center;
        color: #888;
        font-size: 0.95em;
        padding: 1rem 0;
    }
    .landing-footer {
        text-align: center;
        color: #555;
        font-size: 0.8em;
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Access Gate / Landing Page ---
ACCESS_CODE = os.environ.get("ACCESS_CODE", "")
if ACCESS_CODE:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        # --- Hero ---
        st.markdown(
            '<div class="landing-hero">'
            '<div class="landing-logo">JijnasaAI</div>'
            '<div class="landing-tagline">the desire to know</div>'
            '<div class="landing-headline">'
            'Ask GPT-4o, Claude, and Gemini the same question.<br>See which answers best.'
            '</div>'
            '<div class="landing-subheadline">'
            'The AI workspace for people who refuse to be locked into one model.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # --- Value prop cards ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                '<div class="landing-card">'
                '<div class="landing-card-icon">&#9878;</div>'
                '<h3>Compare models side-by-side</h3>'
                '<p>Send one prompt to 2-3 models at once. '
                'Watch them stream answers in parallel. '
                'Pick the best one for the task.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                '<div class="landing-card">'
                '<div class="landing-card-icon">&#128196;</div>'
                '<h3>Upload docs, get cited answers</h3>'
                '<p>Drop in a PDF or research paper. Ask questions. '
                'Get answers with the exact source paragraph '
                'and similarity score.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                '<div class="landing-card">'
                '<div class="landing-card-icon">&#128176;</div>'
                '<h3>See what every answer costs</h3>'
                '<p>Per-message token counts and cost. '
                'Daily budget with a hard cap. '
                'No surprise bills at the end of the month.</p>'
                '</div>',
                unsafe_allow_html=True,
            )

        # --- Audience ---
        st.markdown(
            '<div class="landing-audience">'
            'Built for developers juggling API keys across providers '
            'and researchers who need cited answers from their papers.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # --- Access code input ---
        gate_col1, gate_col2, gate_col3 = st.columns([1, 2, 1])
        with gate_col2:
            st.markdown("**Have an access code?**")
            code = st.text_input(
                "Enter your access code to start",
                type="password",
                key="access_code_input",
                label_visibility="collapsed",
                placeholder="Enter access code",
            )
            if code:
                if code == ACCESS_CODE:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid access code. Check your invite message.")
            st.caption("20 beta spots open. No access code yet? Ask the person who shared this link.")

        # --- Footer ---
        st.markdown(
            '<div class="landing-footer">'
            'JijnasaAI Beta &middot; Multi-model chat &middot; RAG &middot; '
            'Voice &middot; Cost tracking<br>'
            'Built with FastAPI + Streamlit'
            '</div>',
            unsafe_allow_html=True,
        )

        st.stop()

# --- Initialize session state ---
if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "model_id" not in st.session_state:
    st.session_state.model_id = "gpt-4o"
if "use_rag" not in st.session_state:
    st.session_state.use_rag = False
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = False
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = ""
if "compare_mode" not in st.session_state:
    st.session_state.compare_mode = False
if "compare_models" not in st.session_state:
    st.session_state.compare_models = []

# --- Admin dashboard route ---
if st.query_params.get("admin") == "1":
    render_admin_dashboard()
    st.stop()

# --- Sidebar ---
with st.sidebar:
    render_sidebar()
    st.divider()
    render_document_upload()

# --- Main area ---
render_voice_controls()
render_chat()
