import streamlit as st


def render_voice_controls():
    """Voice input (mic) and TTS output controls."""
    col1, col2 = st.columns([6, 1])

    with col2:
        tts_enabled = st.toggle("TTS", value=False, help="Read responses aloud")
        st.session_state.tts_enabled = tts_enabled

    # Voice input using Streamlit's built-in audio_input
    audio_data = st.audio_input(
        "Record a voice message",
        key="voice_input",
    )

    if audio_data is not None:
        # Only transcribe if we haven't already processed this audio
        audio_key = f"audio_processed_{hash(audio_data.getvalue()[:100])}"
        if audio_key not in st.session_state:
            st.session_state[audio_key] = True
            with st.spinner("Transcribing..."):
                try:
                    result = st.session_state.api_client.transcribe_audio(
                        audio_bytes=audio_data.getvalue(),
                        filename="recording.wav",
                    )
                    transcribed_text = result.get("text", "").strip()
                    if transcribed_text:
                        st.info(f"Transcribed: {transcribed_text}")
                        st.session_state.voice_transcription = transcribed_text
                        st.rerun()
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
