import os

import streamlit as st


def render_cost_display():
    """Show cost metrics in the sidebar with progress bar and breakdown."""
    st.markdown("**Cost Tracker**")

    daily_budget = float(os.environ.get("MAX_DAILY_SPEND_USD", "10.0"))

    try:
        # Session cost
        conv_id = st.session_state.get("conversation_id")
        if conv_id:
            summary = st.session_state.api_client.get_cost_summary(conv_id)
            session_cost = summary.get("total_cost_usd", 0)
            st.metric("This Session", f"${session_cost:.4f}")
            st.caption(
                f"{summary.get('total_input_tokens', 0):,} in / "
                f"{summary.get('total_output_tokens', 0):,} out"
            )

        # Global total with progress bar
        global_summary = st.session_state.api_client.get_cost_summary()
        global_cost = global_summary.get("total_cost_usd", 0)
        st.metric("All Sessions", f"${global_cost:.4f}")

        # Daily budget progress
        if daily_budget > 0:
            progress = min(global_cost / daily_budget, 1.0)
            st.progress(progress)
            st.caption(f"${global_cost:.2f} of ${daily_budget:.2f} daily budget")

        # Model breakdown in expander
        breakdown = global_summary.get("breakdown", [])
        if breakdown:
            with st.expander("Cost by model"):
                for entry in breakdown:
                    model = entry.get("model_id", "unknown")
                    cost = entry.get("cost", 0)
                    op = entry.get("operation", "")
                    if cost > 0:
                        st.caption(f"{model} ({op}): ${cost:.6f}")

    except Exception:
        st.caption("Cost data unavailable")
