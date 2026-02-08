from datetime import datetime, timezone

import streamlit as st

MODELS = [
    {"id": "gpt-4o", "name": "GPT-4o", "provider": "OpenAI", "cost_hint": "~$0.01/msg"},
    {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "provider": "OpenAI", "cost_hint": "~$0.001/msg"},
    {"id": "claude-sonnet-4-5-20250929", "name": "Claude Sonnet 4.5", "provider": "Anthropic", "cost_hint": "~$0.02/msg"},
    {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "provider": "Anthropic", "cost_hint": "~$0.005/msg"},
    {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "provider": "Google", "cost_hint": "~$0.001/msg"},
    {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro", "provider": "Google", "cost_hint": "~$0.01/msg"},
    {"id": "sonar-pro", "name": "Sonar Pro", "provider": "Perplexity", "cost_hint": "~$0.02/msg"},
    {"id": "sonar", "name": "Sonar", "provider": "Perplexity", "cost_hint": "~$0.002/msg"},
    {"id": "sonar-reasoning-pro", "name": "Sonar Reasoning Pro", "provider": "Perplexity", "cost_hint": "~$0.01/msg"},
]

PROVIDER_COLORS = {
    "OpenAI": "#10a37f",
    "Anthropic": "#d4a274",
    "Google": "#4285f4",
    "Perplexity": "#1fb8cd",
}


def _provider_badge_html(provider: str) -> str:
    css_class = f"provider-{provider.lower()}"
    return f'<span class="provider-badge {css_class}">{provider}</span>'


def _time_ago(iso_str: str) -> str:
    """Convert ISO datetime string to relative time."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - dt
        seconds = delta.total_seconds()
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            m = int(seconds / 60)
            return f"{m}m ago"
        elif seconds < 86400:
            h = int(seconds / 3600)
            return f"{h}h ago"
        else:
            d = int(seconds / 86400)
            return f"{d}d ago"
    except Exception:
        return ""


def render_sidebar():
    st.markdown(
        '<h1 style="background: linear-gradient(90deg, #667eea, #764ba2); '
        '-webkit-background-clip: text; -webkit-text-fill-color: transparent; '
        'font-size: 1.8em;">JijnasaAI</h1>',
        unsafe_allow_html=True,
    )
    st.caption("The desire to know — multi-model AI chat")

    # --- Model selector ---
    st.markdown("**Model**")
    _WEB_PROVIDERS = {"Perplexity", "Google"}
    model_labels = [
        f"{m['name']}  {m['cost_hint']}  ({m['provider']})"
        + (" | Web" if m["provider"] in _WEB_PROVIDERS else "")
        for m in MODELS
    ]
    model_ids = [m["id"] for m in MODELS]

    current_idx = 0
    if st.session_state.model_id in model_ids:
        current_idx = model_ids.index(st.session_state.model_id)

    selected_idx = st.selectbox(
        "Select model",
        range(len(MODELS)),
        format_func=lambda i: model_labels[i],
        index=current_idx,
        label_visibility="collapsed",
    )
    st.session_state.model_id = model_ids[selected_idx]

    # Show provider badge for selected model
    selected_model = MODELS[selected_idx]
    badge_html = _provider_badge_html(selected_model["provider"])
    if selected_model["provider"] in _WEB_PROVIDERS:
        badge_html += ' <span style="font-size: 0.8em; color: #4CAF50;">Web search enabled</span>'
    st.markdown(badge_html, unsafe_allow_html=True)

    # --- Temperature ---
    st.session_state.temperature = st.slider(
        "Temperature", 0.0, 2.0, st.session_state.temperature, 0.1
    )

    # --- RAG toggle ---
    st.session_state.use_rag = st.toggle(
        "Use documents (RAG)", value=st.session_state.use_rag
    )

    # --- Compare mode toggle ---
    st.session_state.compare_mode = st.toggle(
        "Compare models", value=st.session_state.get("compare_mode", False),
        help="Send the same prompt to multiple models side-by-side",
    )

    if st.session_state.compare_mode:
        compare_options = [f"{m['name']} ({m['provider']})" for m in MODELS]
        selected_compare = st.multiselect(
            "Models to compare",
            options=range(len(MODELS)),
            format_func=lambda i: compare_options[i],
            default=[0, 2] if not st.session_state.get("compare_models") else [],
            max_selections=3,
        )
        st.session_state.compare_models = [model_ids[i] for i in selected_compare]

    st.divider()

    # --- System prompt ---
    with st.expander("System Prompt"):
        new_prompt = st.text_area(
            "Custom system prompt for this conversation",
            value=st.session_state.get("system_prompt", ""),
            height=100,
            label_visibility="collapsed",
        )
        if new_prompt != st.session_state.get("system_prompt", ""):
            st.session_state.system_prompt = new_prompt
            if st.session_state.conversation_id:
                try:
                    st.session_state.api_client.update_system_prompt(
                        st.session_state.conversation_id, new_prompt
                    )
                except Exception:
                    pass

    st.divider()

    # --- Conversation management ---
    st.markdown("**Conversations**")
    if st.button("New Conversation", use_container_width=True, type="primary"):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.session_state.system_prompt = ""
        st.rerun()

    # List existing conversations
    try:
        conversations = st.session_state.api_client.list_conversations()
        for conv in conversations[:30]:
            title = conv.get("title", "Untitled")[:40]
            is_active = st.session_state.conversation_id == conv["id"]
            created = conv.get("created_at", "")
            model = conv.get("model_id", "")
            time_str = _time_ago(created)

            # Find provider for this model
            provider = ""
            for m in MODELS:
                if m["id"] == model:
                    provider = m["provider"]
                    break

            col1, col2 = st.columns([5, 1])
            with col1:
                label = f"**{title}**" if is_active else title
                if st.button(
                    label,
                    key=f"conv_{conv['id']}",
                    use_container_width=True,
                    disabled=is_active,
                ):
                    st.session_state.conversation_id = conv["id"]
                    st.session_state.messages = _load_messages(conv["id"])
                    st.session_state.system_prompt = conv.get("system_prompt", "")
                    st.rerun()
                # Metadata line below the button
                meta_parts = []
                if time_str:
                    meta_parts.append(time_str)
                if provider:
                    meta_parts.append(provider)
                if meta_parts:
                    st.caption(" · ".join(meta_parts))
            with col2:
                if st.button("X", key=f"del_{conv['id']}"):
                    st.session_state.api_client.delete_conversation(conv["id"])
                    if st.session_state.conversation_id == conv["id"]:
                        st.session_state.conversation_id = None
                        st.session_state.messages = []
                    st.rerun()
    except Exception:
        st.caption("Backend not available. Start the FastAPI server.")


def _load_messages(conversation_id: str) -> list[dict]:
    """Load messages from backend and convert to Streamlit format."""
    msgs = st.session_state.api_client.get_messages(conversation_id)
    return [
        {
            "role": m["role"],
            "content": m["content"],
            "used_docs": bool(m.get("used_docs")),
        }
        for m in msgs
        if m["role"] in ("user", "assistant")
    ]
