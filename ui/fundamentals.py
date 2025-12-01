# ui/fundamentals.py

from __future__ import annotations

from datetime import date
from typing import Optional, Dict, Any

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.fundamentals import FundamentalsService


class FundamentalsInterface:
    """
    Fundamentals UI:
    - Lets user select a symbol
    - Shows key valuation and quality metrics
    - Displays 1-year price history chart with proper styling
    - Stores fundamentals in st.session_state.current_fundamentals
      for the chatbot to use.
    """

    def __init__(self):
        self.service = FundamentalsService()

    def render(self):
        self._init_state()
        self._render_controls()
        self._render_summary()

    def _init_state(self):
        if "fund_symbol" not in st.session_state:
            st.session_state.fund_symbol = "AAPL"
        if "current_fundamentals" not in st.session_state:
            st.session_state.current_fundamentals = None

    def _render_controls(self):
        st.markdown(
            """
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:flex-end;
                margin-bottom:0.75rem;">
                <div>
                    <div style="font-size:0.85rem;color:#9ca3af;letter-spacing:0.16em;text-transform:uppercase;">
                        Fundamentals
                    </div>
                    <div style="font-size:0.9rem;color:#e5e7eb;margin-top:0.2rem;">
                        Inspect valuation, profitability, and recent price behaviour for a given symbol.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container():
            col1, col2, col3 = st.columns([1.2, 0.5, 2.0])

            with col1:
                symbol = st.text_input(
                    "Symbol",
                    value=st.session_state.fund_symbol,
                    max_chars=10,
                    help="Example: AAPL, MSFT, TSLA",
                )
                st.session_state.fund_symbol = symbol.strip().upper()

            with col2:
                st.write("")
                st.write("")
                refresh = st.button("Get Fundamentals")
            
            with col3:
                st.write("")

        if refresh:
            self._load_fundamentals()

        if st.session_state.current_fundamentals is None:
            self._load_fundamentals()

    def _load_fundamentals(self):
        symbol = st.session_state.fund_symbol
        if not symbol:
            st.warning("Please enter a symbol.")
            return

        with st.spinner("Fetching fundamentals and price history..."):
            try:
                f = self.service.get_fundamentals(symbol)
                hist = self.service.get_price_history(symbol, period="1y", interval="1d")
            except Exception as e:
                st.session_state.current_fundamentals = None
                st.error(f"Failed to fetch fundamentals: {e}")
                return

        fund_dict = f.to_dict()
        fund_dict["price_history"] = hist.to_dict(orient="list") if not hist.empty else None
        st.session_state.current_fundamentals = fund_dict

    def _render_summary(self):
        fundamentals = st.session_state.get("current_fundamentals")
        if not fundamentals:
            st.markdown(
                """
                <div style="margin-top:1rem;font-size:0.8rem;color:#64748b;">
                    Once fundamentals are loaded, you will see valuation, profitability, and
                    a one-year price chart here. The chat assistant will also draw on these
                    details when interpreting a symbol.
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        name = fundamentals.get("long_name") or fundamentals.get("symbol")
        sector = fundamentals.get("sector") or "N/A"
        industry = fundamentals.get("industry") or "N/A"

        st.markdown(
            f"""
            <div style="margin-top:0.6rem;margin-bottom:0.8rem;">
                <div style="font-size:1.1rem;font-weight:600;color:#e5e7eb;">
                    {name} ({fundamentals.get("symbol","")})
                </div>
                <div style="font-size:0.8rem;color:#9ca3af;margin-top:0.1rem;">
                    Sector: {sector} | Industry: {industry}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns(2)

        with col_left:
            self._render_valuation_block(fundamentals)

        with col_right:
            self._render_quality_block(fundamentals)

        # Add spacing before the chart
        st.markdown(
            """
            <div style="margin-top:2rem;"></div>
            """,
            unsafe_allow_html=True,
        )

        # Price history chart
        hist_dict = fundamentals.get("price_history")
        if hist_dict:
            hist_df = pd.DataFrame(hist_dict)
            if not hist_df.empty and "close" in hist_df.columns:
                hist_df = hist_df.copy()
                hist_df["date"] = pd.to_datetime(hist_df["date"])
                self._render_price_chart(hist_df, fundamentals.get("symbol", ""))
            else:
                st.info("Price history is not available for this symbol.")
        else:
            st.info("Price history is not available for this symbol.")

        desc = fundamentals.get("short_description")
        if desc:
            st.markdown(
                """
                <div style="margin-top:1.5rem;margin-bottom:0.3rem;
                            font-size:0.8rem;color:#9ca3af;letter-spacing:0.16em;
                            text-transform:uppercase;">
                    Business overview
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="font-size:0.8rem;color:#cbd5e1;line-height:1.5;">
                    {desc}
                </div>
                """,
                unsafe_allow_html=True,
            )

    def _render_price_chart(self, hist_df: pd.DataFrame, symbol: str):
        """Render price chart with styling matching the analysis equity curve."""
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=hist_df["date"],
                y=hist_df["close"],
                mode="lines",
                name="Close Price",
                line=dict(color="#22d3ee", width=2),
                fill="tozeroy",
                fillcolor="rgba(34, 211, 238, 0.1)",
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: $%{y:,.2f}<extra></extra>"
            )
        )
        
        fig.update_layout(
            title={
                'text': f"Price History - {symbol}",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 16, 'color': '#e2e8f0'}
            },
            xaxis={
                'title': {'text': 'Date', 'font': {'size': 14, 'color': '#94a3b8'}},
                'showgrid': True,
                'gridcolor': 'rgba(148, 163, 184, 0.1)',
                'tickfont': {'color': '#94a3b8'},
                'linecolor': 'rgba(148, 163, 184, 0.3)',
            },
            yaxis={
                'title': {'text': 'Price ($)', 'font': {'size': 14, 'color': '#94a3b8'}},
                'showgrid': True,
                'gridcolor': 'rgba(148, 163, 184, 0.1)',
                'tickfont': {'color': '#94a3b8'},
                'tickprefix': '$',
                'tickformat': ',.0f',
                'linecolor': 'rgba(148, 163, 184, 0.3)',
            },
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            paper_bgcolor='rgba(15, 23, 42, 0.0)',
            margin=dict(l=80, r=40, t=60, b=60),
            height=400,
            hovermode='x unified',
            showlegend=False,
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _render_valuation_block(self, f: Dict[str, Any]):
        mc = f.get("market_cap")
        pe = f.get("pe_ratio")
        fpe = f.get("forward_pe")
        pb = f.get("pb_ratio")
        dy = f.get("dividend_yield")

        def fmt_billion(x):
            if x is None:
                return "N/A"
            try:
                b = float(x) / 1_000_000_000
                return f"{b:.1f} B"
            except Exception:
                return "N/A"

        def fmt_pct(x):
            if x is None:
                return "N/A"
            try:
                return f"{float(x):.2f}%"
            except Exception:
                return "N/A"

        def fmt_num(x):
            if x is None:
                return "N/A"
            try:
                return f"{float(x):.2f}"
            except Exception:
                return "N/A"

        st.markdown(
            """
            <div style="font-size:0.75rem;color:#9ca3af;letter-spacing:0.12em;
                        text-transform:uppercase;margin-bottom:0.5rem;font-weight:600;">
                Valuation
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        val_data = [
            ("Market Cap", fmt_billion(mc)),
            ("P/E (Trailing)", fmt_num(pe)),
            ("P/E (Forward)", fmt_num(fpe)),
            ("Price / Book", fmt_num(pb)),
            ("Dividend Yield", fmt_pct(dy)),
        ]
        
        for label, value in val_data:
            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between;padding:0.35rem 0;
                            border-bottom:1px solid rgba(148,163,184,0.1);">
                    <span style="font-size:0.8rem;color:#94a3b8;">{label}</span>
                    <span style="font-size:0.8rem;color:#e2e8f0;font-weight:500;">{value}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def _render_quality_block(self, f: Dict[str, Any]):
        pm = f.get("profit_margin")
        om = f.get("operating_margin")
        roe = f.get("return_on_equity")
        roa = f.get("return_on_assets")
        beta = f.get("beta")
        high = f.get("fifty_two_week_high")
        low = f.get("fifty_two_week_low")

        def fmt_pct(x):
            if x is None:
                return "N/A"
            try:
                return f"{float(x):.2f}%"
            except Exception:
                return "N/A"

        def fmt_num(x):
            if x is None:
                return "N/A"
            try:
                return f"{float(x):.2f}"
            except Exception:
                return "N/A"

        def fmt_price(x):
            if x is None:
                return "N/A"
            try:
                return f"{float(x):.2f}"
            except Exception:
                return "N/A"

        st.markdown(
            """
            <div style="font-size:0.75rem;color:#9ca3af;letter-spacing:0.12em;
                        text-transform:uppercase;margin-bottom:0.5rem;font-weight:600;">
                Quality and Risk
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        quality_data = [
            ("Profit Margin", fmt_pct(pm)),
            ("Operating Margin", fmt_pct(om)),
            ("Return on Equity", fmt_pct(roe)),
            ("Return on Assets", fmt_pct(roa)),
            ("Beta", fmt_num(beta)),
            ("52W Range", f"{fmt_price(low)} - {fmt_price(high)}"),
        ]
        
        for label, value in quality_data:
            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between;padding:0.35rem 0;
                            border-bottom:1px solid rgba(148,163,184,0.1);">
                    <span style="font-size:0.8rem;color:#94a3b8;">{label}</span>
                    <span style="font-size:0.8rem;color:#e2e8f0;font-weight:500;">{value}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )