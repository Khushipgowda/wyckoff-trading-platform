# core/rag_model.py
"""
Enhanced RAG engine for the Wyckoff Trading Intelligence Platform.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class RetrievedQA:
    question: str
    answer: str
    label: str
    score: float


class WyckoffRAG:
    """
    RAG engine with analysis capabilities for Wyckoff methodology.
    """

    def __init__(self, qa_df: pd.DataFrame, ngram_range=(1, 2), use_llm: bool = True):
        self.qa_df = self._normalise_columns(qa_df.copy())
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            stop_words="english",
            min_df=1,
        )
        self.doc_matrix = self._fit_corpus()
        self.backtester = None
        self.fundamentals_service = None
        self.use_llm = use_llm
        self.openai_client = None
        
        if use_llm and OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)

    def set_backtester(self, backtester):
        """Set the backtester instance for analysis queries."""
        self.backtester = backtester

    def set_fundamentals_service(self, service):
        """Set the fundamentals service for fundamental queries."""
        self.fundamentals_service = service

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower(): c for c in df.columns}
        q_col = cols.get("questions") or cols.get("question")
        a_col = cols.get("answers") or cols.get("answer")
        l_col = cols.get("label") or cols.get("labels")

        rename_map = {}
        if q_col and q_col != "question":
            rename_map[q_col] = "question"
        if a_col and a_col != "answer":
            rename_map[a_col] = "answer"
        if l_col and l_col != "label":
            rename_map[l_col] = "label"

        df = df.rename(columns=rename_map)

        required = ["question", "answer"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Dataset must contain a '{col}' column.")

        if "label" not in df.columns:
            df["label"] = "General"

        df = df.dropna(subset=["question", "answer"])
        df["question"] = df["question"].astype(str)
        df["answer"] = df["answer"].astype(str)
        df["label"] = df["label"].astype(str)

        return df.reset_index(drop=True)

    def _fit_corpus(self):
        questions = self.qa_df["question"].tolist()
        return self.vectorizer.fit_transform(questions)

    def _detect_analysis_request(self, query: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if user is requesting an analysis.
        Returns: (is_analysis_request, params)
        """
        query_lower = query.lower()
        
        analysis_patterns = [
            r'analy[sz]e\s+(\w+)',
            r'backtest\s+(\w+)',
            r'test\s+(\w+)\s+(?:from|between)',
            r'run\s+(?:analysis|backtest)\s+(?:on|for)\s+(\w+)',
            r'how\s+(?:did|would|does)\s+(\w+)\s+perform',
        ]
        
        date_patterns = [
            r'from\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s+to\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'between\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s+and\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s+to\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'last\s+(\d+)\s+(year|month|day)s?',
            r'past\s+(\d+)\s+(year|month|day)s?',
        ]
        
        symbol = None
        for pattern in analysis_patterns:
            match = re.search(pattern, query_lower)
            if match:
                symbol = match.group(1).upper()
                break
        
        if not symbol:
            if any(word in query_lower for word in ['analyze', 'analysis', 'backtest']):
                symbol_match = re.search(r'\b([A-Z]{1,5})\b', query.upper())
                if symbol_match and symbol_match.group(1) not in ['I', 'A', 'THE', 'AND', 'OR', 'FOR', 'TO', 'FROM', 'WITH', 'WHAT', 'HOW', 'WHY', 'WHEN', 'IS', 'ARE', 'CAN', 'DO']:
                    symbol = symbol_match.group(1)
        
        if not symbol:
            return False, {}
        
        start_date = None
        end_date = None
        
        for pattern in date_patterns[:3]:
            match = re.search(pattern, query_lower)
            if match:
                start_date = match.group(1).replace('/', '-')
                end_date = match.group(2).replace('/', '-')
                break
        
        if not start_date:
            for pattern in date_patterns[3:]:
                match = re.search(pattern, query_lower)
                if match:
                    num = int(match.group(1))
                    unit = match.group(2)
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    if unit == 'year':
                        start_date = (datetime.now() - timedelta(days=365*num)).strftime('%Y-%m-%d')
                    elif unit == 'month':
                        start_date = (datetime.now() - timedelta(days=30*num)).strftime('%Y-%m-%d')
                    else:
                        start_date = (datetime.now() - timedelta(days=num)).strftime('%Y-%m-%d')
                    break
        
        if not start_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        return True, {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
        }

    def _detect_fundamentals_request(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Check if user is requesting fundamentals data.
        Returns: (is_fundamentals_request, symbol)
        """
        query_lower = query.lower()
        
        fundamentals_keywords = [
            'fundamental', 'fundamentals', 'valuation', 'pe ratio', 'p/e',
            'market cap', 'dividend', 'profit margin', 'roe', 'roa',
            'earnings', 'revenue', 'financials', 'financial data',
            'company info', 'stock info', 'get fundamentals', 'show fundamentals',
            'financial info', 'company data', 'stock data'
        ]
        
        if not any(kw in query_lower for kw in fundamentals_keywords):
            return False, None
        
        fundamentals_patterns = [
            r'fundamentals?\s+(?:for|of|on)\s+(\w+)',
            r'(\w+)\s+fundamentals?',
            r'get\s+fundamentals?\s+(?:for|of)?\s*(\w+)',
            r'show\s+fundamentals?\s+(?:for|of)?\s*(\w+)',
            r'valuation\s+(?:for|of)\s+(\w+)',
            r'financials?\s+(?:for|of)\s+(\w+)',
            r'financial\s+(?:data|info)\s+(?:for|of)\s+(\w+)',
            r'(?:for|of)\s+(\w+)',
        ]
        
        symbol = None
        for pattern in fundamentals_patterns:
            match = re.search(pattern, query_lower)
            if match:
                potential_symbol = match.group(1).upper()
                if potential_symbol not in ['FOR', 'OF', 'THE', 'A', 'AN', 'GET', 'SHOW', 'ME', 'ON', 'DATA', 'INFO']:
                    symbol = potential_symbol
                    break
        
        if not symbol:
            # Try to find any stock ticker pattern (1-5 uppercase letters)
            words = query.upper().split()
            for word in words:
                word_clean = re.sub(r'[^A-Z]', '', word)
                if 1 <= len(word_clean) <= 5 and word_clean not in ['I', 'A', 'THE', 'AND', 'OR', 'FOR', 'TO', 'FROM', 'WITH', 'WHAT', 'HOW', 'WHY', 'WHEN', 'IS', 'ARE', 'CAN', 'DO', 'GET', 'SHOW', 'ME', 'OF', 'DATA', 'INFO', 'FINANCIAL']:
                    symbol = word_clean
                    break
        
        return bool(symbol), symbol

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.05) -> List[RetrievedQA]:
        if not query.strip():
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix)[0]
        idx_sorted = np.argsort(-sims)[:top_k]

        results: List[RetrievedQA] = []
        for idx in idx_sorted:
            score = float(sims[idx])
            if score < min_score:
                continue
            row = self.qa_df.iloc[idx]
            results.append(
                RetrievedQA(
                    question=row["question"],
                    answer=row["answer"],
                    label=row.get("label", "General"),
                    score=score,
                )
            )
        return results

    def _is_off_topic(self, query: str) -> bool:
        """Check if query is clearly off-topic or gibberish."""
        query_lower = query.lower().strip()
        
        # Off-topic keywords
        off_topic_keywords = ['weather', 'news', 'sports', 'movie', 'music', 'food', 'recipe', 
                            'joke', 'funny', 'game', 'play', 'song', 'dance', 'cook', 
                            'capital of', 'president of', 'how to cook', 'how to make',
                            'what are you doing', 'how are you', 'where are you', 'who are you',
                            'tell me a', 'sing', 'poem', 'story']
        if any(kw in query_lower for kw in off_topic_keywords):
            return True
        
        # Check for gibberish (no vowels or very short random chars)
        vowels = set('aeiouAEIOU')
        alpha_chars = [c for c in query if c.isalpha()]
        if len(alpha_chars) > 3 and not any(c in vowels for c in alpha_chars):
            return True
        
        # Very short non-greeting queries
        if len(query_lower) < 3 and query_lower not in ['hi', 'hey']:
            return True
            
        return False

    def generate_answer(
        self,
        user_question: str,
        backtest_context: Optional[Dict[str, Any]] = None,
        fundamentals: Optional[Dict[str, Any]] = None,
        top_k: int = 4,
    ) -> Tuple[str, Optional[Dict]]:
        """
        Generate an answer. Handles analysis, fundamentals, and Q&A requests.
        Returns: (answer_text, new_backtest_result or None)
        """
        
        # Check for greetings first
        query_lower = user_question.lower().strip()
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'howdy', 'greetings']
        if query_lower in greetings or any(query_lower.startswith(g + ' ') or query_lower.startswith(g + ',') or query_lower.startswith(g + '!') for g in greetings):
            return "Hello! I'm your Wyckoff trading assistant. I can help you understand Wyckoff methodology, analyze stocks using the Wyckoff strategy, or fetch company fundamentals. What would you like to explore today?", None
        
        # Check for thank you
        if 'thank' in query_lower:
            return "You're welcome! Feel free to ask if you have more questions about Wyckoff methodology or need any stock analysis.", None
        
        # Check for off-topic or gibberish BEFORE RAG
        if self._is_off_topic(user_question):
            return self._fallback_answer(user_question), None
        
        # Check if user is requesting a NEW analysis
        is_analysis_request, params = self._detect_analysis_request(user_question)
        
        if is_analysis_request and self.backtester:
            try:
                result = self.backtester.run_backtest(
                    symbol=params["symbol"],
                    start_date=params["start_date"],
                    end_date=params["end_date"],
                )
                summary = self._generate_analysis_summary(result)
                return summary, result
            except Exception as e:
                return f"I attempted to run the analysis but encountered an error: {str(e)}. Please check the symbol and date range.", None
        
        # Check if user is requesting fundamentals
        is_fund_request, fund_symbol = self._detect_fundamentals_request(user_question)
        
        if is_fund_request and fund_symbol and self.fundamentals_service:
            try:
                fund_data = self.fundamentals_service.get_fundamentals(fund_symbol)
                summary = self._generate_fundamentals_summary(fund_data)
                return summary, None
            except Exception as e:
                return f"I attempted to fetch fundamentals for {fund_symbol} but encountered an error: {str(e)}. Please verify the symbol is valid.", None
        
        # Standard Q&A
        retrieved = self.retrieve(user_question, top_k=top_k)

        if self.openai_client:
            answer = self._generate_llm_answer(user_question, retrieved)
            return answer, None

        if not retrieved:
            return self._fallback_answer(user_question), None

        return self._build_wyckoff_block(retrieved), None

    def _generate_fundamentals_summary(self, fund) -> str:
        """Generate a formatted fundamentals summary."""
        def fmt_billion(x):
            if x is None:
                return "N/A"
            try:
                b = float(x) / 1_000_000_000
                return f"${b:.2f}B"
            except:
                return "N/A"

        def fmt_pct(x):
            if x is None:
                return "N/A"
            try:
                return f"{float(x):.2f}%"
            except:
                return "N/A"

        def fmt_num(x):
            if x is None:
                return "N/A"
            try:
                return f"{float(x):.2f}"
            except:
                return "N/A"
        
        def fmt_price(x):
            if x is None:
                return "N/A"
            try:
                return f"${float(x):.2f}"
            except:
                return "N/A"

        name = fund.long_name or fund.symbol
        
        summary = f"""Fundamentals for {name} ({fund.symbol})

COMPANY OVERVIEW:
Sector: {fund.sector or 'N/A'}
Industry: {fund.industry or 'N/A'}

VALUATION METRICS:
Market Cap: {fmt_billion(fund.market_cap)}
P/E Ratio (Trailing): {fmt_num(fund.pe_ratio)}
P/E Ratio (Forward): {fmt_num(fund.forward_pe)}
Price to Book: {fmt_num(fund.pb_ratio)}
Dividend Yield: {fmt_pct(fund.dividend_yield)}

PROFITABILITY:
Profit Margin: {fmt_pct(fund.profit_margin)}
Operating Margin: {fmt_pct(fund.operating_margin)}
Return on Equity: {fmt_pct(fund.return_on_equity)}
Return on Assets: {fmt_pct(fund.return_on_assets)}

RISK METRICS:
Beta: {fmt_num(fund.beta)}
52-Week High: {fmt_price(fund.fifty_two_week_high)}
52-Week Low: {fmt_price(fund.fifty_two_week_low)}"""

        return summary

    def _generate_analysis_summary(self, result: Dict) -> str:
        """Generate analysis summary."""
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

        outperform = ret > buyhold
        perf_diff = abs(ret - buyhold)

        summary = f"""Wyckoff Strategy Analysis for {symbol}
Period: {start} to {end}

PERFORMANCE COMPARISON:
Wyckoff Strategy Return: {ret:.2f}%
Buy-and-Hold Return: {buyhold:.2f}%

"""
        if outperform:
            summary += f"The Wyckoff strategy outperformed buy-and-hold by {perf_diff:.2f} percentage points.\n\n"
        else:
            summary += f"Buy-and-hold outperformed the Wyckoff strategy by {perf_diff:.2f} percentage points.\n\n"

        summary += f"""RISK METRICS:
Maximum Drawdown: {max_dd:.2f}%
Sharpe Ratio: {sharpe:.2f}

TRADING ACTIVITY:
Total Trades: {num_trades}
Win Rate: {win_rate:.2f}%
Spring Signals Detected: {springs}
Breakout Signals Detected: {breakouts}

NOTE: Buy-and-Hold means simply purchasing the stock at the start date and holding until the end date, without any trading. This serves as a benchmark to compare the active Wyckoff strategy performance."""

        return summary

    def _generate_llm_answer(self, user_question: str, retrieved: List[RetrievedQA]) -> str:
        """Generate answer using LLM with only retrieved Q&A context."""
        
        context = ""
        if retrieved:
            context = "Relevant Wyckoff knowledge:\n"
            for item in retrieved:
                context += f"\nQ: {item.question}\nA: {item.answer}\n"
        
        system_prompt = """You are a Wyckoff trading methodology expert. 
Answer questions about Wyckoff methodology, market phases, trading psychology, and technical analysis.
Be concise and helpful. Do not use emojis. Do not include category labels in your responses.
If the user wants to analyze a stock, tell them they can request it by saying something like "Analyze AAPL from 2023-01-01 to 2024-01-01".
If the user wants fundamentals, tell them they can request it by saying "Get fundamentals for AAPL"."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_question}"}
                ],
                max_tokens=500,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception:
            if not retrieved:
                return self._fallback_answer(user_question)
            return self._build_wyckoff_block(retrieved)

    def _build_wyckoff_block(self, retrieved: List[RetrievedQA]) -> str:
        """Build response from retrieved Q&A pairs without labels."""
        if not retrieved:
            return self._fallback_answer("")
        
        # If only one result, return it directly
        if len(retrieved) == 1:
            return retrieved[0].answer
        
        # For multiple results, use clean bullet points
        answers = []
        for item in retrieved:
            if item.answer not in answers:  # Avoid duplicates
                answers.append(item.answer)
        
        if len(answers) == 1:
            return answers[0]
        
        # Format with compact bullet points (limit to top 3)
        response = "Based on Wyckoff methodology:"
        for answer in answers[:3]:
            response += f"\n• {answer}"
        return response

    def _fallback_answer(self, user_question: str) -> str:
        """Universal fallback for all unrecognized queries."""
        short_query = user_question[:50] + ('...' if len(user_question) > 50 else '')
        return f"I understand you're asking about '{short_query}'. I'm specialized in Wyckoff trading methodology and stock analysis. I can help you with market phase analysis, trading signals, stock backtesting, and company fundamentals. Could you try asking about accumulation/distribution phases, Spring tests, or request a stock analysis like 'Analyze AAPL last year'?"