# core/fundamentals.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import yfinance as yf
import pandas as pd
import datetime as dt


@dataclass
class Fundamentals:
    symbol: str
    long_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None

    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None

    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None

    beta: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None

    short_description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FundamentalsService:
    """
    Wrapper around yfinance to fetch and normalize company fundamentals
    and a small price history. Designed to be:
      - Resistant to missing fields
      - Simple to use from Streamlit and the chatbot
    """

    def __init__(self):
        pass

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def get_fundamentals(self, symbol: str) -> Fundamentals:
        """
        Fetch and normalize fundamentals for the given symbol.
        Returns a Fundamentals dataclass instance.
        """
        symbol = symbol.upper()
        ticker = yf.Ticker(symbol)

        try:
            info = ticker.info  # may be slow, but usually fine for coursework
        except Exception:
            info = {}

        f = Fundamentals(symbol=symbol)

        # Basic identity
        f.long_name = info.get("longName") or info.get("shortName")
        f.sector = info.get("sector")
        f.industry = info.get("industry")

        # Valuation
        f.market_cap = self._safe_float(info.get("marketCap"))
        f.pe_ratio = self._safe_float(info.get("trailingPE"))
        f.forward_pe = self._safe_float(info.get("forwardPE"))
        f.pb_ratio = self._safe_float(info.get("priceToBook"))

        # Dividend
        f.dividend_yield = self._safe_float(info.get("dividendYield"), scale=100.0)

        # Profitability
        f.profit_margin = self._safe_float(info.get("profitMargins"), scale=100.0)
        f.operating_margin = self._safe_float(info.get("operatingMargins"), scale=100.0)
        f.return_on_equity = self._safe_float(info.get("returnOnEquity"), scale=100.0)
        f.return_on_assets = self._safe_float(info.get("returnOnAssets"), scale=100.0)

        # Risk / volatility
        f.beta = self._safe_float(info.get("beta"))

        # Range
        f.fifty_two_week_high = self._safe_float(info.get("fiftyTwoWeekHigh"))
        f.fifty_two_week_low = self._safe_float(info.get("fiftyTwoWeekLow"))

        # Description
        f.short_description = info.get("longBusinessSummary")

        return f

    def get_price_history(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch recent price history for plotting fundamentals chart.
        period examples: "6mo", "1y", "2y"
        interval examples: "1d", "1wk"
        """
        symbol = symbol.upper()
        ticker = yf.Ticker(symbol)

        try:
            hist = ticker.history(period=period, interval=interval)
        except Exception:
            return pd.DataFrame()

        if hist.empty:
            return pd.DataFrame()

        hist = hist.reset_index()

        # Handle multi-level columns if they exist (newer yfinance versions)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.droplevel(1)

        # Build rename mapping based on what columns actually exist
        rename_map = {}
        
        # Check for Date/Datetime column
        if "Date" in hist.columns:
            rename_map["Date"] = "date"
        elif "Datetime" in hist.columns:
            rename_map["Datetime"] = "date"
        
        # Standard OHLCV columns
        if "Open" in hist.columns:
            rename_map["Open"] = "open"
        if "High" in hist.columns:
            rename_map["High"] = "high"
        if "Low" in hist.columns:
            rename_map["Low"] = "low"
        if "Close" in hist.columns:
            rename_map["Close"] = "close"
        if "Volume" in hist.columns:
            rename_map["Volume"] = "volume"
        
        # Only add Adj Close mapping if it exists
        if "Adj Close" in hist.columns:
            rename_map["Adj Close"] = "adj_close"

        hist = hist.rename(columns=rename_map)

        # Create adj_close from close if it doesn't exist
        if "adj_close" not in hist.columns and "close" in hist.columns:
            hist["adj_close"] = hist["close"]

        # Build column list based on what's available
        desired_columns = ["date", "open", "high", "low", "close", "adj_close", "volume"]
        available_columns = [col for col in desired_columns if col in hist.columns]
        
        if not available_columns or "date" not in available_columns:
            return pd.DataFrame()
        
        hist = hist[available_columns]

        return hist

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _safe_float(self, value, scale: float = 1.0) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value) * scale
        except Exception:
            return None