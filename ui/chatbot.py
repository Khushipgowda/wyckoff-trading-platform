# ui/chatbot.py

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

import streamlit as st

from core.rag_model import WyckoffRAG
from core.backtest import WyckoffBacktester
from core.fundamentals import FundamentalsService
from core.transformer_model import TransformerService


class ChatbotInterface:
    """
    Chat interface with dual model support (RAG and Transformer).
    Includes analysis and fundamentals capabilities.
    """

    def __init__(self, rag: WyckoffRAG):
        self.rag = rag
        self.backtester = WyckoffBacktester(initial_capital=10_000.0)
        self.fundamentals_service = FundamentalsService()
        self.rag.set_backtester(self.backtester)
        self.rag.set_fundamentals_service(self.fundamentals_service)
        
        # Initialize Transformer service
        self.transformer_service: Optional[TransformerService] = None
        self._init_transformer()

    def _init_transformer(self):
        """Initialize the Transformer model if available."""
        model_paths = [
            Path("wyckoff_chatbot.pth"),
            Path("models/wyckoff_chatbot.pth"),
            Path("core/wyckoff_chatbot.pth"),
        ]
        
        for path in model_paths:
            if path.exists():
                self.transformer_service = TransformerService(str(path))
                if self.transformer_service.is_loaded:
                    print(f"Transformer model loaded from {path}")
                    return
        
        self.transformer_service = TransformerService()
        print("Transformer model not found. Transformer mode will be unavailable.")

    def render(self):
        self._ensure_state()
        self._apply_chat_styles()
        self._handle_actions()

        sidebar_col, divider_col, chat_col = st.columns([0.28, 0.02, 0.70], gap="small")

        with sidebar_col:
            self._render_sidebar()

        with divider_col:
            st.markdown(
                """<div style="width: 1px; height: 650px; 
                background: linear-gradient(180deg, rgba(148,163,184,0.3) 0%, rgba(148,163,184,0.15) 100%);
                margin: 0 auto;"></div>""",
                unsafe_allow_html=True
            )

        with chat_col:
            self._render_chat_window()

    def _apply_chat_styles(self):
        st.markdown("""
            <style>
            body, html {
                scrollbar-width: thin;
                scrollbar-color: rgba(51,65,85,0.6) rgba(15,23,42,0.3);
            }
            
            [data-testid="stVerticalBlockBorderWrapper"] {
                border: none !important;
                background: transparent !important;
            }
            
            [data-testid="baseButton-secondary"] {
                background: rgba(30,41,59,0.7) !important;
                border: 1px solid rgba(148,163,184,0.5) !important;
                color: #94a3b8 !important;
                border-radius: 10px !important;
                font-size: 0.84rem !important;
                padding: 0.75rem 1rem !important;
                margin-bottom: 0.25rem !important;
            }
            
            [data-testid="baseButton-secondary"]:hover {
                background: rgba(40,51,69,0.9) !important;
                color: #e2e8f0 !important;
            }
            
            .msg-user-container {
                display: flex;
                justify-content: flex-end;
                margin-bottom: 1rem;
                padding-left: 20%;
            }
            
            .msg-user {
                background: linear-gradient(135deg, rgba(34,211,238,0.12), rgba(56,189,248,0.08));
                border: 1px solid rgba(34,211,238,0.22);
                border-radius: 16px 16px 4px 16px;
                padding: 0.8rem 1rem;
            }
            
            .msg-user-inner {
                display: flex;
                align-items: flex-start;
                gap: 0.7rem;
            }
            
            .msg-user-content {
                font-size: 0.875rem;
                line-height: 1.65;
                color: #e2e8f0;
            }
            
            .msg-user-avatar {
                width: 30px;
                height: 30px;
                min-width: 30px;
                border-radius: 50%;
                background: linear-gradient(135deg, #22d3ee, #38bdf8);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.7rem;
                font-weight: 600;
                color: #0b1220;
            }
            
            .msg-user-time {
                font-size: 0.65rem;
                color: #64748b;
                margin-top: 0.35rem;
                text-align: right;
            }
            
            .msg-bot-container {
                display: flex;
                justify-content: flex-start;
                margin-bottom: 1rem;
                padding-right: 10%;
            }
            
            .msg-bot {
                background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.9));
                border: 1px solid rgba(148,163,184,0.18);
                border-radius: 16px 16px 16px 4px;
                padding: 0.8rem 1rem;
                max-width: 100%;
            }
            
            .msg-bot-inner {
                display: flex;
                align-items: flex-start;
                gap: 0.7rem;
            }
            
            .msg-bot-content {
                font-size: 0.875rem;
                line-height: 1.5;
                color: #cbd5e1;
                white-space: pre-line;
            }
            
            .msg-bot-avatar {
                width: 30px;
                height: 30px;
                min-width: 30px;
                border-radius: 50%;
                background: linear-gradient(135deg, #1e293b, #0f172a);
                border: 1px solid rgba(45,212,191,0.35);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.65rem;
                font-weight: 600;
                color: #2dd4bf;
            }
            
            .msg-bot-avatar.transformer {
                border: 1px solid rgba(168,85,247,0.5);
                color: #a855f7;
            }
            
            .msg-bot-time {
                font-size: 0.65rem;
                color: #475569;
                margin-top: 0.35rem;
            }
            
            /* Model Toggle Styles */
            .model-toggle-container {
                margin-top: 1.5rem;
                padding-top: 1rem;
                border-top: 1px solid rgba(148,163,184,0.1);
            }
            
            .model-toggle-label {
                font-size: 0.68rem;
                letter-spacing: 0.15em;
                text-transform: uppercase;
                color: #475569;
                margin-bottom: 0.6rem;
            }
            
            .model-toggle-wrapper {
                display: flex;
                background: rgba(15,23,42,0.6);
                border: 1px solid rgba(148,163,184,0.15);
                border-radius: 8px;
                padding: 4px;
                gap: 4px;
            }
            
            .model-toggle-btn {
                flex: 1;
                padding: 0.5rem 0.75rem;
                border-radius: 6px;
                font-size: 0.75rem;
                font-weight: 500;
                text-align: center;
                cursor: pointer;
                transition: all 0.2s ease;
                border: none;
                background: transparent;
                color: #64748b;
            }
            
            .model-toggle-btn.active {
                background: linear-gradient(135deg, rgba(34,211,238,0.15), rgba(56,189,248,0.1));
                color: #22d3ee;
                border: 1px solid rgba(34,211,238,0.3);
            }
            
            .model-toggle-btn.active.transformer {
                background: linear-gradient(135deg, rgba(168,85,247,0.15), rgba(139,92,246,0.1));
                color: #a855f7;
                border: 1px solid rgba(168,85,247,0.3);
            }
            
            .model-toggle-btn:hover:not(.active) {
                background: rgba(30,41,59,0.5);
                color: #94a3b8;
            }
            
            .model-status {
                font-size: 0.65rem;
                color: #475569;
                margin-top: 0.5rem;
                text-align: center;
            }
            
            .model-status.active-rag {
                color: #2dd4bf;
            }
            
            .model-status.active-transformer {
                color: #a855f7;
            }
            </style>
        """, unsafe_allow_html=True)

    def _ensure_state(self):
        if "chats" not in st.session_state:
            st.session_state.chats = {}

        if "active_chat_id" not in st.session_state or st.session_state.active_chat_id is None:
            self._create_new_chat()

        if st.session_state.active_chat_id not in st.session_state.chats:
            if st.session_state.chats:
                st.session_state.active_chat_id = next(iter(st.session_state.chats.keys()))
            else:
                self._create_new_chat()
        
        if "renaming_chat" not in st.session_state:
            st.session_state.renaming_chat = None
        
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = "rag"

    def _create_new_chat(self, name: Optional[str] = None) -> str:
        chat_id = str(uuid.uuid4())
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        existing_count = len(st.session_state.get("chats", {}))
        default_name = f"Session {existing_count + 1}"

        st.session_state.chats[chat_id] = {
            "id": chat_id,
            "name": name or default_name,
            "created_at": now_str,
            "messages": [],
        }
        st.session_state.active_chat_id = chat_id
        return chat_id

    def _delete_chat(self, chat_id: str):
        if chat_id in st.session_state.chats:
            del st.session_state.chats[chat_id]
        
        if not st.session_state.chats:
            self._create_new_chat()
        elif st.session_state.active_chat_id == chat_id:
            st.session_state.active_chat_id = next(iter(st.session_state.chats.keys()))

    def _delete_all_chats(self):
        st.session_state.chats = {}
        self._create_new_chat()

    def _get_chats_list(self) -> List[Dict]:
        chats = list(st.session_state.chats.values())
        chats.sort(key=lambda c: c.get("created_at", ""), reverse=True)
        return chats

    def _get_active_chat(self) -> Dict:
        chat_id = st.session_state.active_chat_id
        if chat_id not in st.session_state.chats:
            chat_id = self._create_new_chat()
        return st.session_state.chats[chat_id]

    def _append_message(self, role: str, content: str, model_used: str = "rag"):
        chat = self._get_active_chat()
        now_str = datetime.now().strftime("%H:%M")
        chat["messages"].append({
            "role": role,
            "content": content,
            "timestamp": now_str,
            "model": model_used if role == "bot" else None,
        })
        st.session_state.chats[chat["id"]] = chat

    def _handle_actions(self):
        if st.session_state.get("action_new_chat"):
            self._create_new_chat()
            st.session_state.action_new_chat = False
            st.rerun()

        if st.session_state.get("action_delete_all"):
            self._delete_all_chats()
            st.session_state.action_delete_all = False
            st.rerun()

    def _render_sidebar(self):
        st.markdown(
            "<p style='font-size: 0.72rem; letter-spacing: 0.18em; text-transform: uppercase; "
            "color: #64748b; margin-bottom: 0.75rem; font-weight: 500;'>Chat Sessions</p>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("New Chat", key="btn_new", use_container_width=True):
                st.session_state.action_new_chat = True
                st.rerun()
        with col2:
            if st.button("Clear All", key="btn_delete_all", use_container_width=True):
                st.session_state.action_delete_all = True
                st.rerun()

        st.markdown(
            "<div style='height: 1px; background: rgba(148,163,184,0.2); margin: 0.6rem 0 0.75rem 0;'></div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<p style='font-size: 0.68rem; letter-spacing: 0.15em; text-transform: uppercase; "
            "color: #475569; margin-bottom: 0.5rem;'>Try Asking</p>",
            unsafe_allow_html=True
        )
        
        suggestions = [
            ("What is Wyckoff methodology?", "wyckoff"),
            ("Analyze AAPL last year", "analyze"),
            ("Get fundamentals for MSFT", "fund"),
        ]
        
        for text, key in suggestions:
            if st.button(text, key=f"sug_{key}", use_container_width=True):
                self._handle_user_query(text)

        st.markdown(
            "<div style='height: 1px; background: rgba(148,163,184,0.2); margin: 0.6rem 0 0.75rem 0;'></div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<p style='font-size: 0.68rem; letter-spacing: 0.15em; text-transform: uppercase; "
            "color: #475569; margin-bottom: 0.5rem;'>Previous Conversations</p>",
            unsafe_allow_html=True
        )

        chats = self._get_chats_list()
        active_id = st.session_state.active_chat_id

        if chats:
            with st.container(height=200, border=False):
                for chat in chats:
                    cid = chat["id"]
                    name = chat.get("name", "Untitled")
                    is_active = cid == active_id
                    
                    if is_active:
                        st.markdown(
                            f"""<div style="background: linear-gradient(135deg, rgba(34,211,238,0.15), rgba(56,189,248,0.08)); 
                                border-left: 4px solid #22d3ee; 
                                border-radius: 8px; 
                                padding: 0.6rem 0.75rem; 
                                margin-bottom: 0.25rem;">
                                <span style="color: #22d3ee; font-weight: 600; font-size: 0.85rem;">{name}</span>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    else:
                        if st.button(name, key=f"sel_{cid}", use_container_width=True, type="secondary"):
                            st.session_state.active_chat_id = cid
                            st.rerun()

        # Model Toggle Section
        self._render_model_toggle()

    def _render_model_toggle(self):
        """Render the minimal model toggle at the bottom of sidebar."""
        transformer_available = (
            self.transformer_service is not None and 
            self.transformer_service.is_loaded
        )
        
        st.markdown(
            "<div class='model-toggle-container'>"
            "<p class='model-toggle-label'>AI Model</p>"
            "</div>",
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        
        current_model = st.session_state.selected_model
        
        with col1:
            if st.button("RAG", key="model_rag", use_container_width=True):
                st.session_state.selected_model = "rag"
                st.rerun()
        
        with col2:
            if transformer_available:
                if st.button("Transformer", key="model_transformer", use_container_width=True):
                    st.session_state.selected_model = "transformer"
                    st.rerun()
            else:
                st.button("Transformer", key="model_transformer_disabled", use_container_width=True, disabled=True)
        
        # Status indicator
        if current_model == "rag":
            st.markdown(
                "<p style='font-size: 0.65rem; color: #2dd4bf; margin-top: 0.4rem; text-align: center;'>RAG Active</p>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<p style='font-size: 0.65rem; color: #a855f7; margin-top: 0.4rem; text-align: center;'>Transformer Active</p>",
                unsafe_allow_html=True
            )
        
        if not transformer_available:
            st.markdown(
                "<p style='font-size: 0.6rem; color: #64748b; margin-top: 0.3rem; text-align: center; font-style: italic;'>Transformer not loaded</p>",
                unsafe_allow_html=True
            )

    def _render_chat_window(self):
        active_chat = self._get_active_chat()
        chat_id = active_chat.get("id")
        chat_name = active_chat.get("name", "Current Session")
        chat_created = active_chat.get("created_at", "")
        messages = active_chat.get("messages", [])

        if st.session_state.get("renaming_header") == chat_id:
            col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
            with col1:
                new_name = st.text_input("Rename chat", value=chat_name, key="rename_input")
            with col2:
                st.write("")
                st.write("")
                if st.button("Save", key="save_rename_btn"):
                    if new_name.strip():
                        st.session_state.chats[chat_id]["name"] = new_name.strip()
                    st.session_state.renaming_header = None
                    st.rerun()
            with col3:
                st.write("")
                st.write("")
                if st.button("Cancel", key="cancel_rename_btn"):
                    st.session_state.renaming_header = None
                    st.rerun()
        else:
            header_col1, header_col2, header_col3 = st.columns([0.70, 0.15, 0.15])
            
            with header_col1:
                st.markdown(
                    f"<p style='font-size: 1.2rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.2rem;'>{chat_name}</p>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<p style='font-size: 0.78rem; color: #64748b; margin-bottom: 0;'>Started {chat_created}</p>",
                    unsafe_allow_html=True
                )
            
            with header_col2:
                if st.button("Rename", key="header_rename_btn"):
                    st.session_state.renaming_header = chat_id
                    st.rerun()
            
            with header_col3:
                if st.button("Delete", key="header_delete_btn"):
                    self._delete_chat(chat_id)
                    st.rerun()

        st.markdown(
            "<div style='height: 1px; background: rgba(148,163,184,0.2); margin: 0.6rem 0 1rem 0;'></div>",
            unsafe_allow_html=True
        )

        if messages:
            with st.container(height=450, border=False):
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    timestamp = msg.get("timestamp", "")
                    model_used = msg.get("model", "rag")

                    if role == "user":
                        st.markdown(
                            f"""<div class="msg-user-container">
                                <div class="msg-user">
                                    <div class="msg-user-inner">
                                        <div class="msg-user-content">{content}</div>
                                        <div class="msg-user-avatar">U</div>
                                    </div>
                                    <div class="msg-user-time">{timestamp}</div>
                                </div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    else:
                        avatar_class = "msg-bot-avatar transformer" if model_used == "transformer" else "msg-bot-avatar"
                        avatar_text = "TF" if model_used == "transformer" else "AI"
                        
                        st.markdown(
                            f"""<div class="msg-bot-container">
                                <div class="msg-bot">
                                    <div class="msg-bot-inner">
                                        <div class="{avatar_class}">{avatar_text}</div>
                                        <div class="msg-bot-content">{content}</div>
                                    </div>
                                    <div class="msg-bot-time">{timestamp}</div>
                                </div>
                            </div>""",
                            unsafe_allow_html=True
                        )
        else:
            st.markdown(
                """
                <div style="display: flex; justify-content: center; align-items: center; min-height: 450px;">
                    <div style="text-align: center; padding: 2rem 1.5rem; color: #64748b;">
                        <div style="font-size: 1.1rem; color: #94a3b8; margin-bottom: 0.6rem; font-weight: 500;">Start a conversation</div>
                        <div style="font-size: 0.85rem; max-width: 400px; line-height: 1.6;">
                            Ask about Wyckoff methodology, request stock analysis, or get company fundamentals.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        user_input = st.chat_input("Ask about Wyckoff, analyze stocks, or get fundamentals...")

        if user_input and user_input.strip():
            self._handle_user_query(user_input.strip())

    def _handle_user_query(self, query: str):
        """Handle user query with proper model selection and parameters."""
        self._append_message("user", query)
        
        selected_model = st.session_state.selected_model

        try:
            if selected_model == "transformer":
                if self.transformer_service and self.transformer_service.is_loaded:
                    # Use SAME parameters as training script
                    answer = self.transformer_service.generate_response(
                        query,
                        max_len=50,
                        temperature=0.8,
                        top_k=5
                    )
                else:
                    answer = "Transformer model is not loaded. Please add 'wyckoff_chatbot.pth' to the project folder."
                
                self._append_message("bot", answer, model_used="transformer")
            else:
                # RAG model
                answer, new_backtest = self.rag.generate_answer(
                    user_question=query,
                    backtest_context=None,
                    fundamentals=None,
                )
                
                if new_backtest:
                    st.session_state.current_backtest = new_backtest
                
                self._append_message("bot", answer, model_used="rag")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            answer = f"I encountered an issue: {str(e)}"
            self._append_message("bot", answer, model_used=selected_model)

        st.rerun()