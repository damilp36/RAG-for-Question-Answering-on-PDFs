from __future__ import annotations

import streamlit as st


def init_session_state() -> None:
    defaults = {
        "last_answer": "",
        "last_sources": [],
        "selected_scope": "All indexed documents",
        "streaming_enabled": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value