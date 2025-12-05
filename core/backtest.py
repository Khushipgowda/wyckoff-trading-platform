# core/backtest.py

from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: str  # "LONG"
    pnl: float      # profit/loss in currency
    pnl_pct: float  # profit/loss in percent


class WyckoffBacktester:
    """
    Backtester implementing authentic Wyckoff methodology.
    Based on Richard Wyckoff's principles:
    - Phase A/B: Consolidation detection via rolling range
    - Phase C: Spring test (price drops below range, recovers with volume)
    - Phase D: Breakout confirmation above resistance with volume
    """

    def __init__(self, initial_capital: float = 10_000.0, window: int = 40):
        self.initial_capital = initial_capital
        self.window = window  # Rolling window for range detection

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def run_backtest(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        strategy_name: str = "Wyckoff Strategy",
    ) -> Dict:
        """
        Main entry point.
        Returns a dict with:
        - summary metrics
        - equity curve
        - trade list
        - signal details for analysis
        - indicator data for charting
        """
        # Normalize dates (handle both / and - formats)
        start_date = start_date.replace('/', '-')
        end_date = end_date.replace('/', '-')

        df = self._load_price_data(symbol, start_date, end_date)
        if df.empty:
            raise ValueError(f"No price data returned for {symbol}. Please check the symbol and date range.")

        df = self._apply_wyckoff_strategy(df)

        trades, equity_curve = self._simulate_trades(df)

        metrics = self._compute_metrics(trades, equity_curve, df)

        # Count signals for analysis summary
        spring_count = df['spring'].sum() if 'spring' in df.columns else 0
        breakout_count = df['breakout'].sum() if 'breakout' in df.columns else 0
        exit_count = df['exit_signal'].sum() if 'exit_signal' in df.columns else 0

        # Prepare indicator data for charting
        indicator_data = self._prepare_indicator_data(df)

        result = {
            "symbol": symbol.upper(),
            "start": start_date,
            "end": end_date,
            "strategy": "Wyckoff Strategy",
            "return": metrics["total_return_pct"],
            "buyhold_return": metrics.get("buyhold_return_pct", 0),
            "max_drawdown": metrics["max_drawdown_pct"],
            "win_rate": metrics["win_rate_pct"],
            "num_trades": metrics["num_trades"],
            "spring_signals": int(spring_count),
            "breakout_signals": int(breakout_count),
            "exit_signals": int(exit_count),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "equity_curve": equity_curve,
            "trades": [t.__dict__ for t in trades],
            "indicator_data": indicator_data,
            "price_data": df[['date', 'close', 'range_high', 'range_low']].to_dict('records') if 'range_high' in df.columns else None,
        }

        return result

    # ---------------------------------------------------------
    # Prepare indicator data for charting
    # ---------------------------------------------------------
    def _prepare_indicator_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare indicator data for the Wyckoff indicators chart.
        """
        indicator_df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        if 'range_high' in df.columns:
            indicator_df['range_high'] = df['range_high']
        if 'range_low' in df.columns:
            indicator_df['range_low'] = df['range_low']
        if 'vol_avg' in df.columns:
            indicator_df['vol_avg'] = df['vol_avg']
        if 'spring' in df.columns:
            indicator_df['spring'] = df['spring']
        if 'breakout' in df.columns:
            indicator_df['breakout'] = df['breakout']
        if 'exit_signal' in df.columns:
            indicator_df['exit_signal'] = df['exit_signal']
        
        return indicator_df

    # ---------------------------------------------------------
    # Data loader (FIXED)
    # ---------------------------------------------------------
    def _load_price_data(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """
        Loads daily OHLCV data using yfinance.
        Fixed to handle various yfinance quirks and column formats.
        """
        symbol = symbol.upper().strip()
        
        # Ensure dates are in correct format (YYYY-MM-DD)
        start = start.replace('/', '-')
        end = end.replace('/', '-')
        
        try:
            # Method 1: Try using Ticker.history() first - more reliable
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start, end=end, auto_adjust=True)
            
            if data is None or data.empty:
                # Method 2: Fallback to yf.download()
                print(f"Ticker.history() returned empty, trying yf.download()")
                data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
            
            if data is None or data.empty:
                print(f"No data returned for {symbol} from {start} to {end}")
                return pd.DataFrame()
            
            # Handle multi-level columns if they exist (newer yfinance versions)
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten multi-level columns by taking the first level
                data.columns = data.columns.droplevel(1)
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Normalize column names to lowercase
            rename_map = {}
            for col in data.columns:
                col_str = str(col)
                col_lower = col_str.lower().strip()
                
                if col_lower in ('date', 'datetime', 'index'):
                    rename_map[col] = 'date'
                elif col_lower == 'open':
                    rename_map[col] = 'open'
                elif col_lower == 'high':
                    rename_map[col] = 'high'
                elif col_lower == 'low':
                    rename_map[col] = 'low'
                elif col_lower == 'close':
                    rename_map[col] = 'close'
                elif col_lower == 'volume':
                    rename_map[col] = 'volume'
                elif col_lower in ('adj close', 'adj_close', 'adjclose'):
                    rename_map[col] = 'adj_close'
            
            data = data.rename(columns=rename_map)
            
            # Verify we have the date column
            if 'date' not in data.columns:
                # Try to find a datetime column
                for col in data.columns:
                    if pd.api.types.is_datetime64_any_dtype(data[col]):
                        data = data.rename(columns={col: 'date'})
                        break
            
            # Verify we have required columns
            required = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required if col not in data.columns]
            
            if missing:
                print(f"Missing required columns: {missing}")
                print(f"Available columns: {list(data.columns)}")
                return pd.DataFrame()
            
            # Sort by date
            data = data.sort_values("date").reset_index(drop=True)
            
            # Add adj_close if not present (since we use auto_adjust=True)
            if 'adj_close' not in data.columns:
                data['adj_close'] = data['close']
            
            # Ensure numeric types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
            for col in numeric_cols:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Drop rows with NaN values in critical columns
            data = data.dropna(subset=['date', 'open', 'high', 'low', 'close', 'volume'])
            
            if data.empty:
                print(f"Data became empty after cleaning for {symbol}")
                return pd.DataFrame()
            
            print(f"Successfully loaded {len(data)} rows for {symbol}")
            return data
            
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    # ---------------------------------------------------------
    # Wyckoff Strategy Logic
    # ---------------------------------------------------------
    def _apply_wyckoff_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implements authentic Wyckoff methodology:
        - Phase A/B: Rolling range to detect consolidation
        - Phase C: Spring test detection
        - Phase D: Breakout confirmation
        """
        df = df.copy()
        window = self.window

        # Rolling range to detect consolidation (Wyckoff Phase A/B)
        df['range_high'] = df['close'].rolling(window).max()
        df['range_low'] = df['close'].rolling(window).min()
        
        # Volume average for confirmation
        df['vol_avg'] = df['volume'].rolling(window).mean()

        # Phase C: Spring Test Detection
        # Price drops below range low (spring), then recovers above with volume
        df['spring'] = (
            (df['low'] < df['range_low'].shift(1)) &  # Low drops below range
            (df['close'] > df['range_low'].shift(1)) &  # Close recovers above
            (df['volume'] > df['vol_avg'])  # Volume confirmation
        )

        # Phase D: Breakout Confirmation
        # Breakout above resistance with volume
        df['breakout'] = (
            (df['close'] > df['range_high'].shift(1)) &  # Close above range high
            (df['volume'] > df['vol_avg'])  # Volume confirmation
        )

        # Signal logic
        df['signal'] = 0
        df.loc[df['spring'], 'signal'] = 1   # Long entry on spring
        df.loc[df['breakout'], 'signal'] = 1  # Confirm/add long on breakout

        # Exit when price drops back into the range
        df['exit_signal'] = (
            (df['close'] < df['range_high']) & 
            (df['close'].shift(1) > df['range_high'])
        )
        df.loc[df['exit_signal'], 'signal'] = -1

        return df

    # ---------------------------------------------------------
    # Trade simulation
    # ---------------------------------------------------------
    def _simulate_trades(self, df: pd.DataFrame):
        """
        Position management based on Wyckoff signals.
        """
        df = df.copy().reset_index(drop=True)

        position_active = False
        entry_price = 0.0
        entry_date = None
        cash = self.initial_capital
        shares = 0.0

        trades: List[Trade] = []
        equity_curve_rows = []

        for i, row in df.iterrows():
            signal = int(row["signal"]) if not isinstance(row["signal"], (int, np.integer)) else row["signal"]
            price = float(row["close"]) if not isinstance(row["close"], (float, np.floating)) else row["close"]
            date = row["date"]

            # Entry on buy signal (spring or breakout)
            if not position_active and signal == 1:
                position_active = True
                entry_price = price
                entry_date = date
                shares = cash / price
                cash = 0.0

            # Exit on sell signal
            elif position_active and signal == -1:
                exit_price = price
                exit_date = date
                pnl = (exit_price - entry_price) * shares
                pnl_pct = (exit_price - entry_price) / entry_price * 100.0
                cash = shares * exit_price
                shares = 0.0

                trades.append(
                    Trade(
                        entry_date=entry_date,
                        exit_date=exit_date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction="LONG",
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                    )
                )
                position_active = False

            # Calculate current equity
            if position_active:
                current_equity = shares * price
            else:
                current_equity = cash

            equity_curve_rows.append({
                "date": date,
                "equity": current_equity,
            })

        # Close open position at end
        if position_active and shares > 0:
            last_row = df.iloc[-1]
            price = float(last_row["close"]) if not isinstance(last_row["close"], (float, np.floating)) else last_row["close"]
            date = last_row["date"]
            exit_price = price
            exit_date = date
            pnl = (exit_price - entry_price) * shares
            pnl_pct = (exit_price - entry_price) / entry_price * 100.0
            cash = shares * exit_price
            
            trades.append(
                Trade(
                    entry_date=entry_date,
                    exit_date=exit_date,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    direction="LONG",
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
            )
            
            # Update last equity value
            if equity_curve_rows:
                equity_curve_rows[-1]["equity"] = cash

        equity_curve = pd.DataFrame(equity_curve_rows).dropna()
        equity_curve = equity_curve.sort_values("date").reset_index(drop=True)

        return trades, equity_curve

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    def _compute_metrics(self, trades: List[Trade], equity_curve: pd.DataFrame, df: pd.DataFrame) -> Dict:
        if equity_curve.empty:
            return {
                "total_return_pct": 0.0,
                "buyhold_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate_pct": 0.0,
                "num_trades": 0,
                "sharpe_ratio": 0.0,
            }

        start_equity = self.initial_capital
        end_equity = equity_curve["equity"].iloc[-1]
        total_return_pct = (end_equity / start_equity - 1.0) * 100.0

        # Buy and hold return
        if not df.empty and 'close' in df.columns:
            start_price = df['close'].iloc[0]
            end_price = df['close'].iloc[-1]
            buyhold_return_pct = (end_price / start_price - 1.0) * 100.0
        else:
            buyhold_return_pct = 0.0

        # Max drawdown
        equity_series = equity_curve["equity"].values
        roll_max = np.maximum.accumulate(equity_series)
        drawdown = (equity_series - roll_max) / roll_max
        max_drawdown_pct = drawdown.min() * 100.0

        # Win rate
        num_trades = len(trades)
        if num_trades > 0:
            wins = sum(1 for t in trades if t.pnl > 0)
            win_rate_pct = wins / num_trades * 100.0
        else:
            win_rate_pct = 0.0

        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(equity_curve) > 1:
            returns = equity_curve["equity"].pct_change().dropna()
            if returns.std() > 0:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        return {
            "total_return_pct": round(total_return_pct, 2),
            "buyhold_return_pct": round(buyhold_return_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "win_rate_pct": round(win_rate_pct, 2),
            "num_trades": num_trades,
            "sharpe_ratio": round(sharpe_ratio, 2),
        }

    # ---------------------------------------------------------
    # Analysis Summary (for chatbot)
    # ---------------------------------------------------------
    def get_analysis_summary(self, result: Dict) -> str:
        """
        Generate a natural language summary of backtest results.
        """
        symbol = result.get("symbol", "Unknown")
        start = result.get("start", "")
        end = result.get("end", "")
        ret = result.get("return", 0)
        buyhold = result.get("buyhold_return", 0)
        max_dd = result.get("max_drawdown", 0)
        win_rate = result.get("win_rate", 0)
        num_trades = result.get("num_trades", 0)
        springs = result.get("spring_signals", 0)
        breakouts = result.get("breakout_signals", 0)
        sharpe = result.get("sharpe_ratio", 0)

        # Performance comparison
        outperform = ret > buyhold
        perf_diff = abs(ret - buyhold)

        summary = f"""Wyckoff Strategy Analysis for {symbol} ({start} to {end}):

Strategy Performance: {ret:.2f}% return
Buy-and-Hold Performance: {buyhold:.2f}% return
"""
        if outperform:
            summary += f"The Wyckoff strategy outperformed buy-and-hold by {perf_diff:.2f} percentage points.\n"
        else:
            summary += f"Buy-and-hold outperformed the Wyckoff strategy by {perf_diff:.2f} percentage points.\n"

        summary += f"""
Risk Metrics:
- Maximum Drawdown: {max_dd:.2f}%
- Sharpe Ratio: {sharpe:.2f}

Trading Activity:
- Total Trades: {num_trades}
- Win Rate: {win_rate:.2f}%
- Spring Signals Detected: {springs}
- Breakout Signals Detected: {breakouts}

"""
        # Interpretation
        if win_rate >= 60:
            summary += "The strategy showed strong signal accuracy with a high win rate.\n"
        elif win_rate >= 40:
            summary += "The strategy showed moderate signal accuracy.\n"
        else:
            summary += "The strategy had a lower win rate, which may indicate challenging market conditions or the need for parameter optimization.\n"

        if abs(max_dd) > 20:
            summary += "Note: The maximum drawdown was significant, indicating periods of substantial unrealized losses.\n"

        return summary