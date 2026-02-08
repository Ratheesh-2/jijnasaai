import streamlit as st


def render_document_upload():
    """File uploader in the sidebar for RAG documents."""
    st.subheader("Documents")

    uploaded_files = st.file_uploader(
        "Upload for RAG",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload PDF, TXT, or MD files to use as context",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Use a unique key based on filename to avoid re-uploading
            upload_key = f"uploaded_{uploaded_file.name}_{uploaded_file.size}"
            if upload_key not in st.session_state:
                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    try:
                        result = st.session_state.api_client.upload_document(
                            file_bytes=uploaded_file.getvalue(),
                            filename=uploaded_file.name,
                            conversation_id=st.session_state.get("conversation_id"),
                        )
                        st.session_state[upload_key] = result
                        st.success(
                            f"**{result['filename']}** - "
                            f"{result['chunk_count']} chunks"
                        )
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

    # Show existing documents
    try:
        conv_id = st.session_state.get("conversation_id")
        docs = st.session_state.api_client.list_documents(conv_id)
        if docs:
            st.caption(f"{len(docs)} document(s) indexed")
            for doc in docs[:10]:
                size_kb = doc.get("file_size", 0) / 1024
                st.caption(
                    f"  {doc.get('filename', '?')} "
                    f"({doc.get('chunk_count', 0)} chunks"
                    f"{f', {size_kb:.0f}KB' if size_kb > 0 else ''})"
                )
    except Exception:
        pass
