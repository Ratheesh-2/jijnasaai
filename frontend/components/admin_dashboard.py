import os

import streamlit as st


def render_admin_dashboard():
    """Render the analytics admin dashboard.

    Accessible via ?admin=1 query param, protected by ADMIN_PASSWORD env var.
    Shows usage metrics, model popularity, cost breakdown, and feature adoption.
    """
    admin_password = os.environ.get("ADMIN_PASSWORD", "")

    # Gate behind password
    if not admin_password:
        st.warning("ADMIN_PASSWORD env var not set. Admin dashboard disabled.")
        return

    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        st.markdown("## Admin Dashboard")
        st.markdown("Enter the admin password to view analytics.")
        pwd = st.text_input("Admin Password", type="password", key="admin_pwd_input")
        if pwd:
            if pwd == admin_password:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Invalid admin password.")
        return

    # --- Fetch analytics data ---
    st.markdown("## JijnasaAI Analytics Dashboard")

    period = st.selectbox(
        "Time period",
        options=[7, 14, 30, 90],
        format_func=lambda d: f"Last {d} days",
        index=2,
    )

    try:
        data = st.session_state.api_client.get_analytics_summary(days=period)
    except Exception as e:
        st.error(f"Failed to load analytics: {e}")
        return

    totals = data.get("totals", {})

    # --- Top-level metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Conversations", totals.get("conversations", 0))
    col2.metric("Messages", totals.get("messages", 0))
    col3.metric("Total Spend", f"${totals.get('cost_usd', 0):.4f}")
    col4.metric("Active Days", totals.get("active_days", 0))

    col5, col6, col7 = st.columns(3)
    col5.metric("Documents Uploaded", totals.get("documents_uploaded", 0))
    col6.metric("RAG Messages", totals.get("rag_messages", 0))
    total_msgs = totals.get("messages", 0)
    rag_pct = (
        f"{totals.get('rag_messages', 0) / total_msgs * 100:.1f}%"
        if total_msgs > 0
        else "0%"
    )
    col7.metric("RAG Usage Rate", rag_pct)

    st.divider()

    # --- Daily activity chart ---
    messages_per_day = data.get("messages_per_day", [])
    if messages_per_day:
        st.markdown("### Messages Per Day")
        chart_data = {row["date"]: row["count"] for row in messages_per_day}
        st.bar_chart(chart_data)

    # --- Daily spend chart ---
    daily_spend = data.get("daily_spend", [])
    if daily_spend:
        st.markdown("### Daily API Spend")
        spend_data = {row["date"]: row["cost"] for row in daily_spend}
        st.bar_chart(spend_data)

    st.divider()

    # --- Model popularity ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Model Usage (by message count)")
        model_usage = data.get("model_usage", [])
        if model_usage:
            total_model_msgs = sum(m["count"] for m in model_usage)
            for m in model_usage:
                pct = m["count"] / total_model_msgs * 100 if total_model_msgs > 0 else 0
                st.markdown(
                    f"**{m['model_id']}**: {m['count']} calls ({pct:.1f}%)"
                )
                st.progress(pct / 100)
        else:
            st.caption("No model usage data yet")

    with col_right:
        st.markdown("### Model Costs")
        model_costs = data.get("model_costs", [])
        if model_costs:
            for m in model_costs:
                st.markdown(
                    f"**{m['model_id']}**: ${m['total_cost']:.4f} "
                    f"({m['call_count']} calls)"
                )
                st.caption(
                    f"  {m['total_input_tokens']:,} in / "
                    f"{m['total_output_tokens']:,} out tokens"
                )
        else:
            st.caption("No cost data yet")

    st.divider()

    # --- Operations breakdown ---
    operations = data.get("operations", [])
    if operations:
        st.markdown("### Operations Breakdown")
        for op in operations:
            st.markdown(
                f"**{op['operation']}**: {op['count']} calls, "
                f"${op['cost']:.4f}"
            )

    # --- Feature events ---
    feature_events = data.get("feature_events", [])
    if feature_events:
        st.markdown("### Feature Usage Events")
        for fe in feature_events:
            st.markdown(f"**{fe['event_type']}**: {fe['count']} times")

    st.divider()

    # --- Conversations per day ---
    conversations_per_day = data.get("conversations_per_day", [])
    if conversations_per_day:
        st.markdown("### Conversations Per Day")
        conv_data = {row["date"]: row["count"] for row in conversations_per_day}
        st.bar_chart(conv_data)

    # --- Back to chat ---
    if st.button("Back to Chat"):
        st.session_state.admin_authenticated = False
        st.query_params.clear()
        st.rerun()
