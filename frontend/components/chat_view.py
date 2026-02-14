import threading

import streamlit as st

from frontend.components.sidebar import MODELS

SUGGESTED_PROMPTS = [
    "What are the biggest tech stories this week?",
    "Write a Python function to merge two sorted lists",
    "Summarize my uploaded PDF document",
    "Compare the latest iPhone vs Samsung Galaxy",
    "Help me write a professional email",
    "Explain quantum computing in simple terms",
]

# Models that have built-in web search
WEB_SEARCH_PROVIDERS = {"Perplexity", "Google"}


def _get_model_display_name(model_id: str) -> str:
    for m in MODELS:
        if m["id"] == model_id:
            return m["name"]
    return model_id


def _get_model_provider(model_id: str) -> str:
    for m in MODELS:
        if m["id"] == model_id:
            return m["provider"]
    return ""


def _render_web_sources(web_sources: list[dict]):
    """Render web search citations in a styled expander."""
    if not web_sources:
        return
    with st.expander(f"Web Sources ({len(web_sources)})"):
        for src in web_sources:
            url = src.get("url", "")
            title = src.get("title", "Source")
            source_type = src.get("source", "web")
            badge = "Perplexity" if source_type == "perplexity" else "Google"
            if url:
                st.markdown(
                    f"[{title}]({url})  \n"
                    f"<small style='color: #888;'>{badge} | {url[:80]}{'...' if len(url) > 80 else ''}</small>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(f"{title} ({badge})")


def _render_rag_sources(sources: list[dict]):
    """Render RAG document sources in a styled expander."""
    if not sources:
        return
    with st.expander("Document Sources"):
        for src in sources:
            st.caption(
                f"**{src.get('filename', 'Unknown')}** "
                f"(chunk {src.get('chunk_index', '?')}, "
                f"similarity: {src.get('similarity', '?')})"
            )
            st.text(src.get("content_preview", "")[:200])


def render_chat():
    """Render chat history and handle new input."""
    # Show suggested prompts when no messages
    if not st.session_state.messages:
        st.markdown("### What would you like to explore?")
        cols = st.columns(2)
        for i, prompt_text in enumerate(SUGGESTED_PROMPTS):
            with cols[i % 2]:
                if st.button(
                    prompt_text,
                    key=f"suggest_{i}",
                    use_container_width=True,
                ):
                    _handle_user_message(prompt_text)
                    return
        st.markdown("---")

    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show RAG document sources
            if msg.get("sources"):
                _render_rag_sources(msg["sources"])
            # Show web search sources
            if msg.get("web_sources"):
                _render_web_sources(msg["web_sources"])
            if msg.get("used_docs"):
                st.caption("Used document context")
            if msg.get("web_grounded"):
                st.caption("Grounded in web search")

    # Check for voice transcription
    if st.session_state.get("voice_transcription"):
        prompt = st.session_state.pop("voice_transcription")
        _handle_user_message(prompt)
        return

    # Chat input
    if prompt := st.chat_input("Ask anything..."):
        _handle_user_message(prompt)


def _handle_user_message(prompt: str):
    """Process a user message: display it, stream the response, save to state."""
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if comparison mode is active
    if st.session_state.get("compare_mode") and len(st.session_state.get("compare_models", [])) >= 2:
        _handle_comparison(prompt)
    else:
        _handle_single_model(prompt)


def _handle_single_model(prompt: str):
    """Stream response from a single model."""
    model_name = _get_model_display_name(st.session_state.model_id)
    model_provider = _get_model_provider(st.session_state.model_id)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        status_area = st.container()

        # Show thinking indicator with web search hint
        if model_provider in WEB_SEARCH_PROVIDERS:
            response_placeholder.markdown(f"*{model_name} is searching the web...*")
        else:
            response_placeholder.markdown(f"*{model_name} is thinking...*")

        full_response = ""
        sources = []
        web_sources = []
        usage = {}
        conversation_id = st.session_state.conversation_id

        try:
            for event in st.session_state.api_client.stream_chat(
                message=prompt,
                model_id=st.session_state.model_id,
                conversation_id=conversation_id,
                use_rag=st.session_state.use_rag,
                temperature=st.session_state.temperature,
            ):
                evt_type = event["event"]
                data = event["data"]

                if evt_type == "conversation":
                    st.session_state.conversation_id = data.get("conversation_id")

                elif evt_type == "token":
                    full_response += data.get("text", "")
                    response_placeholder.markdown(full_response + " |")

                elif evt_type == "sources":
                    sources = data if isinstance(data, list) else []

                elif evt_type == "web_sources":
                    web_sources = data if isinstance(data, list) else []

                elif evt_type == "usage":
                    usage = data
                    if data.get("conversation_id"):
                        st.session_state.conversation_id = data["conversation_id"]

                elif evt_type == "error":
                    st.error(f"Error: {data.get('error', 'Unknown error')}")
                    return

                elif evt_type == "done":
                    break

            # Finalize display -- never show silent emptiness
            if full_response:
                response_placeholder.markdown(full_response)
            else:
                response_placeholder.markdown(
                    "*No response received from the model. Please try again.*"
                )

            # Show RAG document sources
            if sources:
                _render_rag_sources(sources)

            # Show web search sources
            if web_sources:
                _render_web_sources(web_sources)

            if usage:
                with status_area:
                    cols = st.columns(3)
                    cols[0].caption(f"In: {usage.get('input_tokens', 0):,} tokens")
                    cols[1].caption(f"Out: {usage.get('output_tokens', 0):,} tokens")
                    cols[2].caption(f"Cost: ${usage.get('cost_usd', 0):.6f}")

        except Exception as e:
            st.error(f"Connection error: {e}")
            return

    # Save to session state
    msg_data = {
        "role": "assistant",
        "content": full_response,
        "sources": sources,
        "web_sources": web_sources,
        "used_docs": bool(sources),
        "web_grounded": bool(web_sources),
    }
    st.session_state.messages.append(msg_data)

    # Play TTS if enabled
    if st.session_state.get("tts_enabled") and full_response:
        try:
            audio_bytes = st.session_state.api_client.synthesize_speech(full_response[:4096])
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
        except Exception:
            pass


def _handle_comparison(prompt: str):
    """Stream responses from multiple models side-by-side."""
    compare_models = st.session_state.compare_models[:3]
    num_models = len(compare_models)

    # Log comparison event for analytics
    st.session_state.api_client.log_analytics_event(
        "comparison_mode",
        {"models": compare_models},
    )

    # Create side-by-side columns
    cols = st.columns(num_models)
    placeholders = []
    results = [{"text": "", "usage": {}, "error": None, "web_sources": []} for _ in compare_models]

    for i, model_id in enumerate(compare_models):
        model_name = _get_model_display_name(model_id)
        provider = _get_model_provider(model_id)
        with cols[i]:
            st.markdown(f"**{model_name}**")
            if provider in WEB_SEARCH_PROVIDERS:
                st.caption("Web search enabled")
            placeholders.append(st.empty())
            placeholders[-1].markdown(f"*{model_name} is thinking...*")

    def _stream_model(idx, model_id):
        """Stream a single model response (runs in a thread)."""
        try:
            for event in st.session_state.api_client.stream_chat(
                message=prompt,
                model_id=model_id,
                conversation_id=None,  # Don't save comparison chats
                use_rag=st.session_state.use_rag,
                temperature=st.session_state.temperature,
            ):
                evt_type = event["event"]
                data = event["data"]
                if evt_type == "token":
                    results[idx]["text"] += data.get("text", "")
                elif evt_type == "usage":
                    results[idx]["usage"] = data
                elif evt_type == "web_sources":
                    results[idx]["web_sources"] = data if isinstance(data, list) else []
                elif evt_type == "error":
                    results[idx]["error"] = data.get("error", "Unknown error")
                    return
                elif evt_type == "done":
                    return
        except Exception as e:
            results[idx]["error"] = str(e)

    # Launch threads for each model
    threads = []
    for i, model_id in enumerate(compare_models):
        t = threading.Thread(target=_stream_model, args=(i, model_id))
        t.start()
        threads.append(t)

    # Poll results and update display until all threads complete
    import time
    all_done = False
    while not all_done:
        all_done = all(not t.is_alive() for t in threads)
        for i in range(num_models):
            with cols[i]:
                if results[i]["error"]:
                    placeholders[i].error(results[i]["error"])
                elif results[i]["text"]:
                    placeholders[i].markdown(results[i]["text"] + (" |" if not all_done else ""))
        if not all_done:
            time.sleep(0.1)

    # Show final results with usage and web sources
    for i in range(num_models):
        with cols[i]:
            if results[i]["text"]:
                placeholders[i].markdown(results[i]["text"])
            elif not results[i]["error"]:
                placeholders[i].markdown(
                    "*No response received. Please try again.*"
                )
            # Show web sources per model
            if results[i].get("web_sources"):
                _render_web_sources(results[i]["web_sources"])
            usage = results[i].get("usage", {})
            if usage:
                st.caption(
                    f"In: {usage.get('input_tokens', 0):,} | "
                    f"Out: {usage.get('output_tokens', 0):,} | "
                    f"${usage.get('cost_usd', 0):.6f}"
                )

    # Save the first model's response as the main conversation response
    if results[0]["text"]:
        combined = "\n\n".join(
            f"**{_get_model_display_name(compare_models[i])}:**\n{results[i]['text']}"
            for i in range(num_models)
            if results[i]["text"]
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": combined,
        })
