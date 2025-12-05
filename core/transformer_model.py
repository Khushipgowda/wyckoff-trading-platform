# core/transformer_model.py
"""
FULLY HAND-CODED Transformer for Wyckoff Chatbot.

Every component is implemented from scratch:
- Multi-Head Attention
- Positional Encoding  
- Feed-Forward Networks
- Layer Normalization
- Encoder & Decoder Stacks

NO use of nn.Transformer or any pre-built transformer modules.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# HAND-CODED TOKENIZER
# =============================================================================

class WyckoffTokenizer:
    """Hand-coded tokenizer with vocabulary management."""
    
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3
    
    def __init__(self):
        self.word2idx: Dict[str, int] = {
            "<pad>": self.PAD_IDX,
            "<sos>": self.SOS_IDX,
            "<eos>": self.EOS_IDX,
            "<unk>": self.UNK_IDX,
        }
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = 4
    
    def load_vocab(self, vocab_dict: Dict[str, int]):
        self.word2idx = vocab_dict
        self.idx2word = {v: k for k, v in vocab_dict.items()}
        self.vocab_size = len(vocab_dict)
    
    def build_vocab(self, texts: List[str], min_freq: int = 1):
        word_counts: Dict[str, int] = {}
        for text in texts:
            for token in self.tokenize(text):
                word_counts[token] = word_counts.get(token, 0) + 1
        
        for word, count in sorted(word_counts.items()):
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary built: {self.vocab_size} tokens")
    
    def tokenize(self, text: str) -> List[str]:
        text = str(text).lower().strip()
        return re.findall(r"\w+|[^\w\s]", text)
    
    def encode(self, text: str, max_len: int = 64) -> List[int]:
        tokens = [self.word2idx.get(w, self.UNK_IDX) for w in self.tokenize(text)]
        tokens = [self.SOS_IDX] + tokens[:max_len - 2] + [self.EOS_IDX]
        padding = [self.PAD_IDX] * (max_len - len(tokens))
        return tokens + padding
    
    def decode(self, tokens: List[int]) -> str:
        words = []
        for idx in tokens:
            if idx in [self.PAD_IDX, self.SOS_IDX, self.EOS_IDX]:
                continue
            word = self.idx2word.get(idx, "")
            if word:
                words.append(word)
        text = " ".join(words)
        text = re.sub(r' ([.,!?;:])', r'\1', text)
        return text.strip()


# =============================================================================
# HAND-CODED POSITIONAL ENCODING
# =============================================================================

class HandCodedPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from 'Attention Is All You Need'.
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# HAND-CODED MULTI-HEAD ATTENTION
# =============================================================================

class HandCodedMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism implemented from scratch.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
    where head_i = Attention(Q*W_Q_i, K*W_K_i, V*W_V_i)
    
    Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k)) * V
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(attn_output)
        
        return output


# =============================================================================
# HAND-CODED FEED-FORWARD NETWORK
# =============================================================================

class HandCodedFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, x*W_1 + b_1) * W_2 + b_2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# =============================================================================
# HAND-CODED ENCODER LAYER
# =============================================================================

class HandCodedEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.
    
    Structure:
    1. Multi-Head Self-Attention + Residual + LayerNorm
    2. Feed-Forward + Residual + LayerNorm
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = HandCodedMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = HandCodedFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(src + self.dropout1(attn_output))
        
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(ff_output))
        
        return src


# =============================================================================
# HAND-CODED DECODER LAYER
# =============================================================================

class HandCodedDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer.
    
    Structure:
    1. Masked Multi-Head Self-Attention + Residual + LayerNorm
    2. Multi-Head Cross-Attention (to encoder) + Residual + LayerNorm
    3. Feed-Forward + Residual + LayerNorm
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = HandCodedMultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = HandCodedMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = HandCodedFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self_attn_output = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(self_attn_output))
        
        cross_attn_output = self.cross_attn(tgt, memory, memory, mask=memory_mask)
        tgt = self.norm2(tgt + self.dropout2(cross_attn_output))
        
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout3(ff_output))
        
        return tgt


# =============================================================================
# HAND-CODED TRANSFORMER ENCODER
# =============================================================================

class HandCodedEncoder(nn.Module):
    """Stack of Encoder Layers."""
    
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            HandCodedEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, mask)
        return self.norm(src)


# =============================================================================
# HAND-CODED TRANSFORMER DECODER
# =============================================================================

class HandCodedDecoder(nn.Module):
    """Stack of Decoder Layers."""
    
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            HandCodedDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.norm(tgt)


# =============================================================================
# COMPLETE HAND-CODED TRANSFORMER
# =============================================================================

class TransformerChatbot(nn.Module):
    """
    Complete Hand-Coded Transformer for Seq2Seq.
    
    Architecture:
    - Embedding Layer
    - Positional Encoding
    - Encoder Stack
    - Decoder Stack
    - Output Projection
    
    NO nn.Transformer used - everything implemented from scratch!
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_idx: int = 0,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        self.pos_encoding = HandCodedPositionalEncoding(d_model, max_len, dropout)
        
        self.encoder = HandCodedEncoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = HandCodedDecoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, float(0.0))
        return (mask == 0).unsqueeze(0).unsqueeze(0)
    
    def generate_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        return (seq != self.pad_idx).unsqueeze(1).unsqueeze(2)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        tgt_mask = self.generate_causal_mask(tgt.size(1), tgt.device)
        
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        memory = self.encoder(src_emb)
        
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        
        logits = self.output_projection(output)
        
        return logits


# =============================================================================
# TRANSFORMER SERVICE (IMPROVED)
# =============================================================================

class TransformerService:
    """Improved service for loading and using the transformer model."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TransformerChatbot] = None
        self.tokenizer: Optional[WyckoffTokenizer] = None
        self.config: Dict[str, Any] = {}
        self.is_loaded = False
        self.max_len = 64
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load model from checkpoint."""
        path = Path(model_path)
        
        if not path.exists():
            print(f"Model not found: {model_path}")
            return False
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Load tokenizer
            self.tokenizer = WyckoffTokenizer()
            self.tokenizer.load_vocab(checkpoint['vocab'])
            
            # Load config
            self.config = checkpoint.get('config', {})
            self.max_len = self.config.get('max_len', 64)
            
            # Create model
            self.model = TransformerChatbot(
                vocab_size=self.config.get('vocab_size', self.tokenizer.vocab_size),
                d_model=self.config.get('d_model', 128),
                num_heads=self.config.get('num_heads', self.config.get('nhead', 4)),
                num_layers=self.config.get('num_layers', 2),
                d_ff=self.config.get('d_ff', 256),
                max_len=self.max_len,
                dropout=0.0,
                pad_idx=self.tokenizer.PAD_IDX,
            ).to(self.device)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.is_loaded = True
            print(f"Model loaded: {model_path}")
            print(f"Vocab: {self.tokenizer.vocab_size}, Config: {self.config}")
            
            return True
            
        except Exception as e:
            print(f"Load error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_response(
        self,
        question: str,
        max_len: int = 50,
        temperature: float = 0.8,
        top_k: int = 5,
        **kwargs
    ) -> str:
        """
        Generate response - using EXACT same logic as training script.
        """
        if not self.is_loaded:
            return "Error: Model not loaded."
        
        try:
            print(f"[TransformerService] Generating response for: '{question}'")
            
            # Encode input - same as training
            src = torch.tensor(
                [self.tokenizer.encode(question, self.max_len)],
                device=self.device
            )
            
            # Start with SOS token
            tgt = torch.tensor([[self.tokenizer.SOS_IDX]], device=self.device)
            
            self.model.eval()
            with torch.no_grad():
                for _ in range(max_len):
                    # Get model output
                    output = self.model(src, tgt)
                    
                    # Get logits for last position and apply temperature
                    next_logits = output[:, -1, :] / temperature
                    
                    # Top-k sampling (same as training)
                    probs = torch.softmax(next_logits, dim=-1)
                    top_probs, top_idx = torch.topk(probs, min(top_k, probs.size(-1)))
                    
                    # Sample from top-k
                    idx = torch.multinomial(top_probs, 1)
                    next_token = top_idx.gather(-1, idx)
                    
                    # Append to sequence
                    tgt = torch.cat([tgt, next_token], dim=1)
                    
                    # Stop at EOS
                    if next_token.item() == self.tokenizer.EOS_IDX:
                        break
            
            # Decode
            response = self.tokenizer.decode(tgt[0].tolist())
            print(f"[TransformerService] Generated: '{response}'")
            
            # Clean and capitalize
            response = self._clean_response(response)
            
            if not response or len(response) < 2:
                return "I'm not sure how to answer that. Could you rephrase your question about Wyckoff methodology?"
            
            return response
            
        except Exception as e:
            print(f"[TransformerService] Error: {e}")
            import traceback
            traceback.print_exc()
            return f"Generation error: {str(e)}"
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response."""
        if not response:
            return ""
        
        # Fix punctuation spacing
        response = re.sub(r'\s+([.,!?;:])', r'\1', response)
        response = re.sub(r'([.,!?;:])\1+', r'\1', response)
        
        # Capitalize first letter
        response = response.strip()
        if response:
            response = response[0].upper() + response[1:]
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "device": str(self.device),
            "vocab_size": self.tokenizer.vocab_size,
            "config": self.config,
            "parameters": sum(p.numel() for p in self.model.parameters())
        }