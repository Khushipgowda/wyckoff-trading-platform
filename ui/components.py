# ui/components.py

from typing import List, Dict, Optional
import streamlit as st

# -------------------------------------------------------------
# GLOBAL THEME / STYLES
# -------------------------------------------------------------
def apply_premium_theme():
    css = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0, #020617 45%, #020617 100%);
        color: #e5e7eb;
    }

    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        max-width: 100%;
    }

    .workspace-root {
        max-width: 1280px;
        margin: 0.5rem auto 1.5rem auto;
        padding: 0 2.5rem 1.5rem 2.5rem;
    }

    /* Glass Panels */
    .glass-panel {
        border-radius: 1.25rem;
        background: radial-gradient(circle at top left,
            rgba(15,23,42,0.92),
            rgba(15,23,42,0.98));
        border: 1px solid rgba(148,163,184,0.2);
        padding: 0.9rem 1rem;
    }

    .glass-panel-subtle {
        border-radius: 1.25rem;
        background: radial-gradient(circle at top left,
            rgba(15,23,42,0.8),
            rgba(15,23,42,0.95));
        border: 1px solid rgba(148,163,184,0.16);
        padding: 1rem 1.25rem;
    }

    /* Chat List */
    .chat-list-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.55rem;
    }
    .chat-list-title {
        font-size: 0.8rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #9ca3af;
    }
    .chat-list-items {
        display: flex;
        flex-direction: column;
        gap: 0.3rem;
        max-height: calc(100vh - 140px);
        overflow-y: auto;
    }
    .chat-list-item {
        border-radius: 0.85rem;
        padding: 0.5rem 0.6rem;
        border: 1px solid rgba(31,41,55,0.8);
        background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(15,23,42,0.85));
        cursor: pointer;
    }
    .chat-list-item.active {
        border-color: #22d3ee;
        background: radial-gradient(circle at top left,
            rgba(8,47,73,0.9),
            rgba(15,23,42,0.9));
    }

    /* Chat Bubbles */
    .chat-message {
        display: flex;
        gap: 0.55rem;
        margin-bottom: 0.4rem;
        align-items: flex-start;
    }
    .chat-avatar {
        width: 28px;
        height: 28px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
    }
    .chat-message.user .chat-avatar {
        background: linear-gradient(135deg, #22d3ee, #38bdf8);
        color: #0b1220;
    }
    .chat-message.bot .chat-avatar {
        background: radial-gradient(circle at top left, #0f172a, #020617);
        border: 1px solid rgba(148,163,184,0.45);
    }
    .chat-bubble {
        max-width: 100%;
        padding: 0.5rem 0.7rem;
        border-radius: 0.9rem;
        font-size: 0.82rem;
    }
    .chat-message.user .chat-bubble {
        background: radial-gradient(circle at top left, #0f172a, #020617);
        border: 1px solid rgba(148,163,184,0.45);
    }
    .chat-message.bot .chat-bubble {
        background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(15,23,42,0.9));
        border: 1px solid rgba(45,212,191,0.45);
    }

    /* Metrics */
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-bottom: 0.8rem;
    }
    .metric-card {
        flex: 1 1 120px;
        min-width: 120px;
        border-radius: 0.9rem;
        padding: 0.6rem 0.7rem;
        border: 1px solid rgba(148,163,184,0.25);
        background: radial-gradient(circle at top left,
            rgba(15,23,42,0.85),
            rgba(15,23,42,0.97));
    }
    .metric-label {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e5e7eb;
    }
    .metric-sub {
        font-size: 0.7rem;
        color: #64748b;
        margin-top: 0.1rem;
    }
    
    /* Chat window styles */
    .chat-window-header {
        margin-bottom: 1rem;
    }
    .chat-window-title {
        font-size: 1rem;
        font-weight: 600;
        color: #e5e7eb;
    }
    .chat-window-subtitle {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 0.2rem;
    }
    .chat-transcript {
        margin-bottom: 1rem;
        min-height: 300px;
    }
    .chat-input-row {
        position: relative;
    }
    .chat-meta {
        font-size: 0.65rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# -------------------------------------------------------------
# TOP NAVBAR
# -------------------------------------------------------------
def render_top_nav(active_page: str):
    """
    Render a clickable navigation bar with buttons styled as text links.
    """
    
    # CSS to style buttons as navbar items
    st.markdown("""
        <style>
            /* Hide default Streamlit elements */
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Create fixed navbar background */
            .navbar-bg {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 68px;
                background: rgba(10,14,26,0.94);
                border-bottom: 1px solid rgba(255,255,255,0.08);
                backdrop-filter: blur(12px);
                z-index: 998;
            }
            
            /* Style the container with navigation */
            section.main > div:first-child {
                position: fixed !important;
                top: 0;
                left: 0;
                right: 0;
                height: 68px;
                z-index: 999;
                background: transparent;
            }
            
            /* Style columns container */
            div[data-testid="column"] {
                display: flex;
                align-items: center;
                height: 68px;
            }
            
            /* Brand title */
            .navbar-brand {
                font-size: 19px;
                font-weight: 600;
                color: #e5e7eb;
                letter-spacing: 0.5px;
                padding-left: 8px;
            }
            
            .navbar-brand-accent {
                color: #3b82f6;
                text-shadow: 0 0 12px rgba(59, 130, 246, 0.5), 0 0 24px rgba(59, 130, 246, 0.3);
            }
            
            /* Style navigation buttons to look like text */
            div[data-testid="column"] button {
                background: transparent !important;
                background-color: transparent !important;
                border: none !important;
                outline: none !important;
                color: #b5b9c9 !important;
                font-size: 16px !important;
                font-weight: normal !important;
                padding: 8px 20px !important;
                padding-bottom: 10px !important;
                margin: 0 !important;
                border-bottom: 2px solid transparent !important;
                border-radius: 0 !important;
                transition: all 0.2s !important;
                cursor: pointer !important;
                height: auto !important;
                width: auto !important;
                line-height: normal !important;
                box-shadow: none !important;
                -webkit-box-shadow: none !important;
                -moz-box-shadow: none !important;
            }
            
            /* Remove all button backgrounds and borders */
            div[data-testid="column"] button::before,
            div[data-testid="column"] button::after {
                display: none !important;
            }
            
            div[data-testid="column"] button:hover {
                color: white !important;
                background: transparent !important;
                background-color: transparent !important;
                border: none !important;
                outline: none !important;
                border-bottom: 2px solid rgba(96,165,250,0.6) !important;
                box-shadow: none !important;
            }
            
            div[data-testid="column"] button:active,
            div[data-testid="column"] button:focus,
            div[data-testid="column"] button:focus:not(:active),
            div[data-testid="column"] button:focus-visible {
                background: transparent !important;
                background-color: transparent !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
                color: #b5b9c9 !important;
                border-bottom: 2px solid transparent !important;
            }
            
            /* Override Streamlit's button styling completely */
            .stButton > button {
                background-color: transparent !important;
                background-image: none !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            .stButton > button:hover {
                background-color: transparent !important;
                background-image: none !important;
                border: none !important;
                outline: none !important;
            }
            
            .stButton > button:active,
            .stButton > button:focus {
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            /* Active page text */
            .nav-active {
                color: white;
                font-size: 16px;
                padding: 8px 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #60a5fa;
                display: inline-block;
            }
            
            /* Remove button container styling */
            div[data-testid="column"] > div {
                width: auto !important;
            }
            
            div[data-testid="column"] .stButton {
                width: auto !important;
                min-width: auto !important;
            }
            
            /* Remove any button background colors and borders */
            [data-testid="baseButton-secondary"] {
                background-color: transparent !important;
                background: transparent !important;
                border: none !important;
                outline: none !important;
            }
            
            /* Content offset */
            section.main > div:nth-child(2) {
                margin-top: 84px;
            }
            
            .main .block-container {
                padding-top: 84px !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Create navbar background
    st.markdown('<div class="navbar-bg"></div>', unsafe_allow_html=True)
    
    # Navigation pages
    pages = [
        ("home", "Home"),
        ("chat", "Chat Assistant"),
        ("trading", "Analysis"),
        ("fundamentals", "Fundamentals"),
    ]
    
    # Create navbar with columns - brand on left, nav centered
    nav_cols = st.columns([2.5, 1.0, 1.2, 1.0, 1.2, 2.5])
    
    # Brand
    with nav_cols[0]:
        st.markdown('<div class="navbar-brand">WYCKOFF <span class="navbar-brand-accent">INTELLIGENCE</span></div>', unsafe_allow_html=True)
    
    # Navigation items centered
    for idx, (key, label) in enumerate(pages):
        with nav_cols[idx + 1]:
            if active_page == key:
                # Active page - show as text with underline
                st.markdown(f'<div class="nav-active">{label}</div>', unsafe_allow_html=True)
            else:
                # Clickable button
                if st.button(label, key=f"nav_{key}"):
                    st.session_state.current_page = key
                    st.rerun()

# -------------------------------------------------------------
# CHAT SIDEBAR
# -------------------------------------------------------------
def render_chat_sidebar(chats: List[Dict], active_chat_id: Optional[str]):
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="chat-list-header">
            <div class="chat-list-title">SESSIONS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("New Chat", key="btn_new_chat"):
            st.session_state.trigger_new_chat = True
    with cols[1]:
        if st.button("Clear All", key="btn_clear_chats"):
            st.session_state.trigger_clear_chats = True

    st.markdown('<div class="chat-list-items">', unsafe_allow_html=True)

    for chat in chats:
        cid = chat["id"]
        name = chat.get("name", "Untitled")
        created_at = chat.get("created_at", "")
        active = cid == active_chat_id

        item_class = "chat-list-item active" if active else "chat-list-item"

        if st.button(name, key=f"chat_select_{cid}"):
            st.session_state.active_chat_id = cid

        st.markdown(
            f"""
            <div class="{item_class}">
                <div class="chat-list-item-name">{name}</div>
                <div class="chat-list-item-meta">{created_at}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div></div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# CHAT BUBBLE
# -------------------------------------------------------------
def render_chat_message(role: str, content: str, meta: Optional[str] = None):
    role_class = "user" if role == "user" else "bot"
    avatar_text = "U" if role == "user" else "AI"

    st.markdown(
        f"""
        <div class="chat-message {role_class}">
            <div class="chat-avatar">{avatar_text}</div>
            <div>
                <div class="chat-bubble">{content}</div>
                {f'<div class="chat-meta">{meta}</div>' if meta else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------
# CHAT WINDOW HEADER
# -------------------------------------------------------------
def render_chat_window_header(title: str, subtitle: Optional[str] = None):
    st.markdown('<div class="chat-window-header">', unsafe_allow_html=True)

    left, right = st.columns([3, 1])
    with left:
        st.markdown(f'<div class="chat-window-title">{title}</div>', unsafe_allow_html=True)
        if subtitle:
            st.markdown(f'<div class="chat-window-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    with right:
        st.write("")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# METRICS ROW
# -------------------------------------------------------------
def render_metric_row(metrics: Dict[str, Dict[str, str]]):
    st.markdown('<div class="metric-row">', unsafe_allow_html=True)

    for key, m in metrics.items():
        label = m.get("label", key)
        value = m.get("value", "")
        sub = m.get("sub", "")

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                {f'<div class="metric-sub">{sub}</div>' if sub else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)