# ui/layout.py

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from core.data_loader import load_wyckoff_dataset
from core.rag_model import WyckoffRAG
from ui.components import (
    apply_premium_theme,
    render_top_nav,
)
from ui.chatbot import ChatbotInterface
from ui.trading import TradingInterface
from ui.fundamentals import FundamentalsInterface


def _init_workspace_state():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"

    if "rag" not in st.session_state:
        df = load_wyckoff_dataset()
        st.session_state.rag = WyckoffRAG(df)

    if "chat_interface" not in st.session_state:
        st.session_state.chat_interface = ChatbotInterface(st.session_state.rag)
    if "trading_interface" not in st.session_state:
        st.session_state.trading_interface = TradingInterface()
    if "fundamentals_interface" not in st.session_state:
        st.session_state.fundamentals_interface = FundamentalsInterface()

    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "active_chat_id" not in st.session_state:
        pass


def _get_active_chat():
    chats = st.session_state.get("chats", {})
    active_id = st.session_state.get("active_chat_id")
    if not active_id or active_id not in chats:
        return None, None
    return active_id, chats[active_id]


def _render_overview():
    """Clean professional overview page with Wyckoff intro"""
    
    st.markdown("""
        <style>
        .main .block-container {
            padding-top: 2rem !important;
        }
        
        .core-features-heading {
            text-align: center;
            color: #94a3b8;
            font-size: 18px;
            letter-spacing: 0.3em;
            text-transform: uppercase;
            font-weight: 600;
            margin: 55px 0 30px 0;
        }
        
        .feature-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 24px;
            margin-top: 15px;
        }
        .feature-box {
            background: linear-gradient(135deg, rgba(15,23,42,0.6) 0%, rgba(30,41,59,0.4) 100%);
            border: 1px solid rgba(148,163,184,0.12);
            border-radius: 16px;
            padding: 40px 36px;
            height: 280px;
            transition: all 0.3s ease;
            display: flex;
            flex-direction: column;
        }
        .feature-box:nth-child(1) {
            border-left: 3px solid #10b981;
        }
        .feature-box:nth-child(2) {
            border-left: 3px solid #60a5fa;
        }
        .feature-box:nth-child(3) {
            border-left: 3px solid #a78bfa;
        }
        .feature-box:hover {
            border-color: rgba(77, 163, 255, 0.3);
            transform: translateY(-4px);
        }
        .feature-title {
            font-size: 24px;
            font-weight: 600;
            color: #e5e7eb;
            margin-bottom: 20px;
            letter-spacing: -0.01em;
        }
        .feature-text {
            font-size: 15px;
            color: #94a3b8;
            line-height: 1.8;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Intro section
    st.markdown('''
        <div style="max-width: 920px; margin: 10px auto 0 auto; text-align: center; padding: 0 20px;">
            <div style="font-size: 12px; letter-spacing: 0.3em; text-transform: uppercase; color: #60a5fa; margin-bottom: 20px; font-weight: 500;">
                Institutional-Grade Analysis
            </div>
            <div style="font-size: 36px; font-weight: 600; color: #f1f5f9; line-height: 1.25; margin-bottom: 24px; letter-spacing: -0.02em;">
                Master the Markets with Wyckoff Methodology
            </div>
            <div style="font-size: 16px; color: #94a3b8; line-height: 1.85; max-width: 720px; margin: 0 auto;">
                Richard D. Wyckoff developed a time-tested approach to understanding market behavior through the lens of supply and demand. His method reveals the footprints of institutional traders, helping you identify accumulation and distribution phases before major price movements occur.
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Quote box
    st.markdown('''
        <div style="max-width: 920px; margin: 40px auto 0 auto; padding: 0 20px;">
            <div style="background: linear-gradient(145deg, rgba(15,23,42,0.6) 0%, rgba(30,41,59,0.4) 100%); border: 1px solid rgba(148,163,184,0.1); border-left: 3px solid #22d3ee; border-right: 3px solid #22d3ee; border-radius: 12px; padding: 36px 44px; position: relative; box-shadow: 0 4px 24px rgba(0,0,0,0.15);">
                <div style="position: absolute; top: 20px; left: 28px; font-size: 64px; color: rgba(34,211,238,0.12); font-family: Georgia, serif; line-height: 1; pointer-events: none;">"</div>
                <div style="font-size: 18px; color: #e2e8f0; line-height: 1.75; font-style: italic; text-align: center; padding: 0 24px; position: relative; z-index: 1;">
                    The market is made by the minds of men, and the fluctuations in the market are the result of the mental attitude of those who make the market.
                </div>
                <div style="display: flex; align-items: center; justify-content: center; margin-top: 20px; gap: 12px;">
                    <div style="width: 32px; height: 1px; background: linear-gradient(90deg, transparent, rgba(34,211,238,0.4));"></div>
                    <div style="font-size: 13px; color: #64748b; font-weight: 500; letter-spacing: 0.05em;">
                        Richard D. Wyckoff
                    </div>
                    <div style="width: 32px; height: 1px; background: linear-gradient(90deg, rgba(34,211,238,0.4), transparent);"></div>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Wyckoff Market Phases Section
    st.markdown('''
        <div style="max-width: 1100px; margin: 50px auto 0 auto; padding: 0 20px;">
            <div style="text-align: center; margin-bottom: 32px;">
                <div style="font-size: 18px; letter-spacing: 0.3em; text-transform: uppercase; color: #94a3b8; margin-bottom: 12px; font-weight: 600;">
                    Foundation
                </div>
                <div style="font-size: 20px; font-weight: 600; color: #e2e8f0; letter-spacing: -0.01em;">
                    Four Phases of Market Structure
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px;">
                <div style="background: linear-gradient(145deg, rgba(15,23,42,0.5) 0%, rgba(30,41,59,0.3) 100%); border: 1px solid rgba(148,163,184,0.08); border-top: 2px solid #10b981; border-radius: 10px; padding: 24px 20px;">
                    <div style="font-size: 11px; letter-spacing: 0.15em; text-transform: uppercase; color: #10b981; margin-bottom: 8px; font-weight: 600;">Phase 1</div>
                    <div style="font-size: 17px; font-weight: 600; color: #e2e8f0; margin-bottom: 10px;">Accumulation</div>
                    <div style="font-size: 13px; color: #94a3b8; line-height: 1.6;">Smart money quietly builds positions while price moves sideways after a downtrend.</div>
                </div>
                <div style="background: linear-gradient(145deg, rgba(15,23,42,0.5) 0%, rgba(30,41,59,0.3) 100%); border: 1px solid rgba(148,163,184,0.08); border-top: 2px solid #60a5fa; border-radius: 10px; padding: 24px 20px;">
                    <div style="font-size: 11px; letter-spacing: 0.15em; text-transform: uppercase; color: #60a5fa; margin-bottom: 8px; font-weight: 600;">Phase 2</div>
                    <div style="font-size: 17px; font-weight: 600; color: #e2e8f0; margin-bottom: 10px;">Markup</div>
                    <div style="font-size: 13px; color: #94a3b8; line-height: 1.6;">Demand exceeds supply as price advances with increasing public participation.</div>
                </div>
                <div style="background: linear-gradient(145deg, rgba(15,23,42,0.5) 0%, rgba(30,41,59,0.3) 100%); border: 1px solid rgba(148,163,184,0.08); border-top: 2px solid #f59e0b; border-radius: 10px; padding: 24px 20px;">
                    <div style="font-size: 11px; letter-spacing: 0.15em; text-transform: uppercase; color: #f59e0b; margin-bottom: 8px; font-weight: 600;">Phase 3</div>
                    <div style="font-size: 17px; font-weight: 600; color: #e2e8f0; margin-bottom: 10px;">Distribution</div>
                    <div style="font-size: 13px; color: #94a3b8; line-height: 1.6;">Institutions offload holdings to retail buyers as price consolidates at highs.</div>
                </div>
                <div style="background: linear-gradient(145deg, rgba(15,23,42,0.5) 0%, rgba(30,41,59,0.3) 100%); border: 1px solid rgba(148,163,184,0.08); border-top: 2px solid #ef4444; border-radius: 10px; padding: 24px 20px;">
                    <div style="font-size: 11px; letter-spacing: 0.15em; text-transform: uppercase; color: #ef4444; margin-bottom: 8px; font-weight: 600;">Phase 4</div>
                    <div style="font-size: 17px; font-weight: 600; color: #e2e8f0; margin-bottom: 10px;">Markdown</div>
                    <div style="font-size: 13px; color: #94a3b8; line-height: 1.6;">Supply overwhelms demand as price declines, completing the market cycle.</div>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Key Principles Section
    st.markdown('''
        <div style="max-width: 1100px; margin: 50px auto 0 auto; padding: 0 20px;">
            <div style="text-align: center; margin-bottom: 32px;">
                <div style="font-size: 20px; font-weight: 600; color: #e2e8f0; letter-spacing: -0.01em;">
                    Core Wyckoff Principles
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;">
                <div style="background: linear-gradient(145deg, rgba(15,23,42,0.4) 0%, rgba(30,41,59,0.25) 100%); border: 1px solid rgba(148,163,184,0.08); border-radius: 10px; padding: 28px 24px;">
                    <div style="font-size: 16px; font-weight: 600; color: #e2e8f0; margin-bottom: 12px;">The Composite Man</div>
                    <div style="font-size: 14px; color: #94a3b8; line-height: 1.7;">Visualize the market as controlled by a single entity who accumulates, marks up, distributes, and marks down positions. Understanding his motives reveals market direction.</div>
                </div>
                <div style="background: linear-gradient(145deg, rgba(15,23,42,0.4) 0%, rgba(30,41,59,0.25) 100%); border: 1px solid rgba(148,163,184,0.08); border-radius: 10px; padding: 28px 24px;">
                    <div style="font-size: 16px; font-weight: 600; color: #e2e8f0; margin-bottom: 12px;">Law of Supply and Demand</div>
                    <div style="font-size: 14px; color: #94a3b8; line-height: 1.7;">Price rises when demand exceeds supply and falls when supply exceeds demand. Volume analysis reveals the true balance of power between buyers and sellers.</div>
                </div>
                <div style="background: linear-gradient(145deg, rgba(15,23,42,0.4) 0%, rgba(30,41,59,0.25) 100%); border: 1px solid rgba(148,163,184,0.08); border-radius: 10px; padding: 28px 24px;">
                    <div style="font-size: 16px; font-weight: 600; color: #e2e8f0; margin-bottom: 12px;">Law of Cause and Effect</div>
                    <div style="font-size: 14px; color: #94a3b8; line-height: 1.7;">The magnitude of a price move is directly proportional to the accumulation or distribution that precedes it. Longer consolidations yield larger moves.</div>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Core Features heading
    st.markdown("""
        <div class="core-features-heading">PLATFORM FEATURES</div>
    """, unsafe_allow_html=True)
    
    
    st.markdown("""
        <div class="feature-container">
            <div class="feature-box">
                <div class="feature-title">Analysis</div>
                <div class="feature-text">
                    Backtest the Wyckoff strategy on any stock symbol. 
                    The system detects Spring tests and Breakout confirmations using volume analysis.
                    Compare strategy performance against buy-and-hold benchmarks.
                </div>
            </div>
            <div class="feature-box">
                <div class="feature-title">AI Assistant</div>
                <div class="feature-text">
                    RAG-powered chat trained on Wyckoff methodology.
                    Ask questions about market phases, run analyses directly from chat, 
                    or get interpretations of your backtest results.
                </div>
            </div>
            <div class="feature-box">
                <div class="feature-title">Fundamentals</div>
                <div class="feature-text">
                    Access real-time financial data including valuation ratios, 
                    profitability metrics, and price history.
                    Combine fundamental analysis with Wyckoff technical insights.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def _render_chat_page():
    chat_interface = st.session_state.get("chat_interface")
    if chat_interface:
        chat_interface.render()
    else:
        st.error("Chat interface not initialized")


def _render_trading_page():
    trading_interface = st.session_state.get("trading_interface")
    if trading_interface:
        trading_interface.render()
    else:
        st.error("Trading interface not initialized")


def _render_fundamentals_page():
    fundamentals_interface = st.session_state.get("fundamentals_interface")
    if fundamentals_interface:
        fundamentals_interface.render()
    else:
        st.error("Fundamentals interface not initialized")


def render_workspace():
    _init_workspace_state()
    apply_premium_theme()

    allowed_pages = {"home", "chat", "trading", "fundamentals"}
    if st.session_state.current_page not in allowed_pages:
        st.session_state.current_page = "home"

    active_page = st.session_state.current_page
    render_top_nav(active_page=active_page)

    st.markdown('<div class="workspace-root">', unsafe_allow_html=True)

    if active_page == "home":
        _render_overview()
    elif active_page == "chat":
        _render_chat_page()
    elif active_page == "trading":
        _render_trading_page()
    elif active_page == "fundamentals":
        _render_fundamentals_page()
    else:
        _render_overview()

    st.markdown("</div>", unsafe_allow_html=True)