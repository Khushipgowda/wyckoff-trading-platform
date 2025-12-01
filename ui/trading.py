# ui/trading.py

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, Dict, Any

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.backtest import WyckoffBacktester
from ui.components import render_metric_row


class TradingInterface:
    """
    Stock Analysis UI with Wyckoff strategy backtesting.
    """

    def __init__(self):
        self.backtester = WyckoffBacktester(initial_capital=10_000.0)

    def render(self):
        self._apply_styles()
        self._init_state()
        self._render_controls()
        self._render_results()

    def _apply_styles(self):
        """Apply custom styles for the expander."""
        st.markdown("""
            <style>
            /* Style the expander to match theme */
            .streamlit-expanderHeader {
                background: rgba(30, 41, 59, 0.5) !important;
                border: 1px solid rgba(148, 163, 184, 0.2) !important;
                border-radius: 8px !important;
                color: #94a3b8 !important;
            }
            
            .streamlit-expanderHeader:hover {
                background: rgba(30, 41, 59, 0.8) !important;
                border-color: rgba(148, 163, 184, 0.4) !important;
                color: #e2e8f0 !important;
            }
            
            .streamlit-expanderContent {
                background: rgba(15, 23, 42, 0.5) !important;
                border: 1px solid rgba(148, 163, 184, 0.15) !important;
                border-top: none !important;
                border-radius: 0 0 8px 8px !important;
            }
            
            /* Fix expander styling */
            [data-testid="stExpander"] {
                background: transparent !important;
                border: none !important;
            }
            
            [data-testid="stExpander"] > div:first-child {
                background: rgba(30, 41, 59, 0.5) !important;
                border: 1px solid rgba(148, 163, 184, 0.2) !important;
                border-radius: 8px !important;
            }
            
            [data-testid="stExpander"] > div:first-child:hover {
                background: rgba(30, 41, 59, 0.8) !important;
                border-color: rgba(148, 163, 184, 0.4) !important;
            }
            
            [data-testid="stExpander"] summary {
                color: #94a3b8 !important;
            }
            
            [data-testid="stExpander"] summary:hover {
                color: #e2e8f0 !important;
            }
            </style>
        """, unsafe_allow_html=True)

    def _init_state(self):
        if "trading_symbol" not in st.session_state:
            st.session_state.trading_symbol = "AAPL"
        if "trading_start" not in st.session_state:
            st.session_state.trading_start = date.today() - timedelta(days=365)
        if "trading_end" not in st.session_state:
            st.session_state.trading_end = date.today()
        if "current_backtest" not in st.session_state:
            st.session_state.current_backtest = None
        if "date_error" not in st.session_state:
            st.session_state.date_error = None

    def _validate_dates(self, start_dt: date, end_dt: date) -> tuple[bool, str]:
        """Validate date inputs."""
        today = date.today()
        
        if start_dt >= end_dt:
            return False, "Start date must be before end date"
        
        if start_dt > today:
            return False, "Start date cannot be in the future"
        
        if end_dt > today:
            return False, "End date cannot be in the future"
        
        if (end_dt - start_dt).days < 30:
            return False, "Date range must be at least 30 days"
        
        if (end_dt - start_dt).days > 365 * 10:
            return False, "Date range cannot exceed 10 years"
        
        return True, ""

    def _render_controls(self):
        st.markdown(
            """
            <div style="margin-bottom:0.75rem;">
                <div style="font-size:0.85rem;color:#9ca3af;letter-spacing:0.16em;text-transform:uppercase;">
                    Stock Analysis
                </div>
                <div style="font-size:0.9rem;color:#e5e7eb;margin-top:0.2rem;">
                    Backtest the Wyckoff strategy and compare against Buy-and-Hold benchmark.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container():
            col1, col2, col3, col4 = st.columns([1.2, 1.1, 1.1, 1.0])

            with col1:
                symbol = st.text_input(
                    "Symbol",
                    value=st.session_state.trading_symbol,
                    max_chars=10,
                    help="Enter a stock ticker (e.g., AAPL, MSFT, GOOGL)",
                )
                st.session_state.trading_symbol = symbol.strip().upper()

            with col2:
                start_dt = st.date_input(
                    "Start Date",
                    value=st.session_state.trading_start,
                    max_value=date.today(),
                )
                st.session_state.trading_start = start_dt

            with col3:
                end_dt = st.date_input(
                    "End Date",
                    value=st.session_state.trading_end,
                    max_value=date.today(),
                )
                st.session_state.trading_end = end_dt

            with col4:
                st.write("")
                st.write("")
                run_btn = st.button("Run Analysis", use_container_width=True)

        # Show date validation error if any
        if st.session_state.date_error:
            st.markdown(
                f"""<div style="background: linear-gradient(135deg, rgba(251,191,36,0.1), rgba(245,158,11,0.05));
                    border: 1px solid rgba(251,191,36,0.3); border-radius: 8px; padding: 0.6rem 1rem;
                    margin: 0.5rem 0; font-size: 0.85rem; color: #fbbf24;">
                    {st.session_state.date_error}
                </div>""",
                unsafe_allow_html=True,
            )

        # Strategy explanation - simplified, no buy-and-hold explanation
        with st.expander("About Wyckoff Strategy", expanded=False):
            st.markdown(
                """
                **How it works:**
                
                The strategy uses a 40-day rolling window to detect Wyckoff phases and generates signals based on:
                
                - **Spring Test:** Price drops below the range low (support), then recovers above it with above-average volume. This indicates absorption of selling pressure.
                
                - **Breakout Confirmation:** Price breaks above the range high (resistance) with above-average volume. This confirms the start of a markup phase.
                
                - **Exit Signal:** Price drops back into the consolidation range after a breakout, indicating potential trend weakness.
                """,
            )

        if run_btn:
            self._run_backtest()

    def _run_backtest(self):
        symbol = st.session_state.trading_symbol
        start_dt = st.session_state.trading_start
        end_dt = st.session_state.trading_end

        st.session_state.date_error = None

        if not symbol:
            st.session_state.date_error = "Please enter a stock symbol"
            st.rerun()
            return

        is_valid, error_msg = self._validate_dates(start_dt, end_dt)
        if not is_valid:
            st.session_state.date_error = error_msg
            st.rerun()
            return

        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        with st.spinner("Running Wyckoff analysis..."):
            try:
                result = self.backtester.run_backtest(
                    symbol=symbol,
                    start_date=start_str,
                    end_date=end_str,
                )
                st.session_state.current_backtest = result
                st.session_state.date_error = None
            except ValueError as e:
                error_msg = str(e)
                if "No price data" in error_msg:
                    st.session_state.date_error = f"No data available for {symbol}. Please check the symbol is valid."
                else:
                    st.session_state.date_error = f"Unable to run analysis: {error_msg}"
                st.session_state.current_backtest = None
            except Exception as e:
                st.session_state.date_error = f"Analysis error: {str(e)}"
                st.session_state.current_backtest = None

    def _render_results(self):
        backtest = st.session_state.get("current_backtest")
        if not backtest:
            if not st.session_state.get("date_error"):
                st.markdown(
                    """
                    <div style="margin-top:1rem;font-size:0.85rem;color:#64748b; line-height: 1.6;">
                        Enter a symbol and date range, then click <strong>Run Analysis</strong> to see results.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            return

        # Metrics
        strategy_return = backtest.get('return', 0)
        buyhold_return = backtest.get('buyhold_return', 0)
        outperformed = strategy_return > buyhold_return
        
        metrics = {
            "ret": {
                "label": "Wyckoff Strategy",
                "value": f"{strategy_return:.2f}%",
                "sub": "Active Trading",
            },
            "buyhold": {
                "label": "Buy-and-Hold",
                "value": f"{buyhold_return:.2f}%",
                "sub": "Benchmark",
            },
            "dd": {
                "label": "Max Drawdown",
                "value": f"{backtest.get('max_drawdown', 0):.2f}%",
                "sub": "",
            },
            "wr": {
                "label": "Win Rate",
                "value": f"{backtest.get('win_rate', 0):.2f}%",
                "sub": f"{backtest.get('num_trades', 0)} trades",
            },
        }
        render_metric_row(metrics)

        # Performance comparison message
        diff = strategy_return - buyhold_return
        if outperformed:
            st.markdown(
                f"""<div style="background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(34,197,94,0.05));
                    border: 1px solid rgba(16,185,129,0.3); border-radius: 8px; padding: 0.75rem 1rem;
                    margin: 0.5rem 0 1rem 0; font-size: 0.85rem; color: #10b981;">
                    Strategy outperformed buy-and-hold by {diff:.2f} percentage points
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div style="background: linear-gradient(135deg, rgba(251,191,36,0.1), rgba(245,158,11,0.05));
                    border: 1px solid rgba(251,191,36,0.3); border-radius: 8px; padding: 0.75rem 1rem;
                    margin: 0.5rem 0 1rem 0; font-size: 0.85rem; color: #fbbf24;">
                    Buy-and-hold outperformed strategy by {abs(diff):.2f} percentage points
                </div>""",
                unsafe_allow_html=True,
            )

        # Signal statistics
        springs = backtest.get('spring_signals', 0)
        breakouts = backtest.get('breakout_signals', 0)
        sharpe = backtest.get('sharpe_ratio', 0)
        
        st.markdown(
            f"""
            <div style="display: flex; gap: 1.5rem; margin-bottom: 1rem;">
                <div style="font-size: 0.8rem; color: #94a3b8;">
                    <span style="color: #22d3ee;">Spring Signals:</span> {springs}
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8;">
                    <span style="color: #60a5fa;">Breakout Signals:</span> {breakouts}
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8;">
                    <span style="color: #a78bfa;">Sharpe Ratio:</span> {sharpe:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Equity curve
        eq_df = backtest.get("equity_curve")
        symbol = backtest.get("symbol", "")
        
        if isinstance(eq_df, pd.DataFrame) and not eq_df.empty:
            self._render_equity_chart(eq_df, symbol)

        # Trade history
        trades = backtest.get("trades", [])
        st.markdown(
            """
            <div style="margin-top:0.75rem;margin-bottom:0.25rem;
                        font-size:0.8rem;color:#9ca3af;letter-spacing:0.16em;
                        text-transform:uppercase;">
                Trade History
            </div>
            """,
            unsafe_allow_html=True,
        )

        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"]).dt.date
            trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"]).dt.date
            trades_df["pnl"] = trades_df["pnl"].round(2)
            trades_df["pnl_pct"] = trades_df["pnl_pct"].round(2)
            trades_df["entry_price"] = trades_df["entry_price"].round(2)
            trades_df["exit_price"] = trades_df["exit_price"].round(2)

            st.dataframe(
                trades_df[
                    ["entry_date", "exit_date", "direction", "entry_price", "exit_price", "pnl", "pnl_pct"]
                ].rename(columns={
                    "entry_date": "Entry",
                    "exit_date": "Exit",
                    "direction": "Type",
                    "entry_price": "Entry ($)",
                    "exit_price": "Exit ($)",
                    "pnl": "P&L ($)",
                    "pnl_pct": "Return (%)",
                }),
                use_container_width=True,
            )
        else:
            st.info("No trades generated. Market conditions may not have triggered Wyckoff signals.")

    def _render_equity_chart(self, eq_df: pd.DataFrame, symbol: str):
        """Render equity curve with proper axis labels."""
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=eq_df["date"],
                y=eq_df["equity"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#22d3ee", width=2),
                fill="tozeroy",
                fillcolor="rgba(34, 211, 238, 0.1)",
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: $%{y:,.2f}<extra></extra>"
            )
        )
        
        fig.update_layout(
            title={
                'text': f"Equity Curve - {symbol}",
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
                'title': {'text': 'Portfolio Value ($)', 'font': {'size': 14, 'color': '#94a3b8'}},
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