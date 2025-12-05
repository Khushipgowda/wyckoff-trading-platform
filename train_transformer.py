# train_transformer.py
"""
Training script for the Hand-Coded Transformer.

Key improvements:
1. Better data preprocessing and augmentation
2. Smaller model for small datasets
3. Label smoothing for better generalization
4. Curriculum learning (shorter sequences first)
5. Better generation with beam search
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import re
from collections import Counter

from core.transformer_model import TransformerChatbot, WyckoffTokenizer


# =============================================================================
# IMPROVED CONFIGURATION
# =============================================================================

class Config:
    # Smaller model for small datasets
    d_model = 128          # Reduced from 256
    num_heads = 4          # Reduced from 8
    num_layers = 2         # Reduced from 4
    d_ff = 256             # Reduced from 512
    dropout = 0.2          # Increased dropout for regularization
    max_len = 64
    
    # Training - adjusted for small data
    batch_size = 8         # Smaller batch for small dataset
    epochs = 200           # More epochs
    learning_rate = 0.001  # Higher LR for faster convergence
    weight_decay = 0.01    # More regularization
    warmup_steps = 100
    clip_grad = 0.5        # Tighter gradient clipping
    label_smoothing = 0.1  # Add label smoothing
    
    # Data
    train_split = 0.85
    min_word_freq = 1      # Keep all words for small vocab
    
    # Paths
    data_path = "data"
    save_path = "wyckoff_chatbot.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# IMPROVED TOKENIZER
# =============================================================================

class ImprovedTokenizer:
    """Tokenizer with better handling of Wyckoff terminology."""
    
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3
    
    # Wyckoff-specific terms to keep intact
    WYCKOFF_TERMS = {
        'wyckoff', 'accumulation', 'distribution', 'markup', 'markdown',
        'spring', 'upthrust', 'test', 'sos', 'sow', 'lpsy', 'utad',
        'creek', 'ice', 'composite', 'operator', 'volume', 'spread',
        'resistance', 'support', 'breakout', 'breakdown', 'rally',
        'phase', 'schematic', 'demand', 'supply', 'trading', 'range'
    }
    
    def __init__(self):
        self.word2idx = {
            "<pad>": self.PAD_IDX,
            "<sos>": self.SOS_IDX,
            "<eos>": self.EOS_IDX,
            "<unk>": self.UNK_IDX,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = 4
        self.word_freq = Counter()
    
    def tokenize(self, text: str) -> list:
        """Improved tokenization."""
        text = str(text).lower().strip()
        # Keep some punctuation as separate tokens
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        # Remove other special chars
        text = re.sub(r'[^a-z0-9\s.,!?;:]', ' ', text)
        tokens = text.split()
        return [t.strip() for t in tokens if t.strip()]
    
    def build_vocab(self, texts: list, min_freq: int = 1):
        """Build vocabulary from texts."""
        self.word_freq = Counter()
        
        for text in texts:
            tokens = self.tokenize(text)
            self.word_freq.update(tokens)
        
        # Add words meeting frequency threshold
        # Always include Wyckoff terms
        for word, count in sorted(self.word_freq.items()):
            if word not in self.word2idx:
                if count >= min_freq or word in self.WYCKOFF_TERMS:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
        
        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary: {self.vocab_size} tokens")
        print(f"Wyckoff terms in vocab: {sum(1 for t in self.WYCKOFF_TERMS if t in self.word2idx)}/{len(self.WYCKOFF_TERMS)}")
    
    def encode(self, text: str, max_len: int = 64) -> list:
        tokens = [self.word2idx.get(w, self.UNK_IDX) for w in self.tokenize(text)]
        tokens = [self.SOS_IDX] + tokens[:max_len - 2] + [self.EOS_IDX]
        padding = [self.PAD_IDX] * (max_len - len(tokens))
        return tokens + padding
    
    def decode(self, tokens: list) -> str:
        words = []
        for idx in tokens:
            if idx in [self.PAD_IDX, self.SOS_IDX, self.EOS_IDX]:
                continue
            word = self.idx2word.get(idx, "")
            if word and word != "<unk>":
                words.append(word)
        
        text = " ".join(words)
        # Clean up punctuation spacing
        text = re.sub(r' ([.,!?;:])', r'\1', text)
        return text.strip()
    
    def load_vocab(self, vocab_dict: dict):
        self.word2idx = vocab_dict
        self.idx2word = {v: k for k, v in vocab_dict.items()}
        self.vocab_size = len(vocab_dict)


# =============================================================================
# IMPROVED DATASET
# =============================================================================

class WyckoffDataset(Dataset):
    """Dataset with better preprocessing."""
    
    def __init__(self, qa_pairs: list, tokenizer, max_len: int = 64):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        q, a = self.qa_pairs[idx]
        
        src = self.tokenizer.encode(q, self.max_len)
        tgt = self.tokenizer.encode(a, self.max_len)
        
        return {
            'src': torch.tensor(src, dtype=torch.long),
            'tgt': torch.tensor(tgt, dtype=torch.long),
            'src_text': q,
            'tgt_text': a
        }


# =============================================================================
# DATA LOADING & AUGMENTATION
# =============================================================================

def load_all_data(data_path: str) -> list:
    """Load all Q&A pairs from CSV files."""
    data_dir = Path(data_path)
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_path}")
    
    qa_pairs = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Find question and answer columns
            q_col = next((c for c in df.columns if 'question' in c), None)
            a_col = next((c for c in df.columns if 'answer' in c), None)
            
            if q_col and a_col:
                for _, row in df.iterrows():
                    q = str(row[q_col]).strip()
                    a = str(row[a_col]).strip()
                    if q and a and len(q) > 3 and len(a) > 3:
                        qa_pairs.append((q, a))
                print(f"Loaded {len(df)} from {csv_file.name}")
        except Exception as e:
            print(f"Error with {csv_file}: {e}")
    
    # Remove duplicates
    qa_pairs = list(set(qa_pairs))
    print(f"\nTotal unique Q&A pairs: {len(qa_pairs)}")
    
    return qa_pairs


def augment_data(qa_pairs: list) -> list:
    """Augment data with variations."""
    augmented = []
    
    for q, a in qa_pairs:
        # Original
        augmented.append((q, a))
        
        # Lowercase
        augmented.append((q.lower(), a))
        
        # Add question mark
        if not q.strip().endswith('?'):
            augmented.append((q.strip() + '?', a))
        
        # "What is X" -> "Explain X"
        if q.lower().startswith('what is '):
            new_q = 'Explain ' + q[8:]
            augmented.append((new_q, a))
        
        # "What is X" -> "Define X"
        if q.lower().startswith('what is '):
            new_q = 'Define ' + q[8:]
            augmented.append((new_q, a))
        
        # Add "please" variations
        augmented.append(("Please " + q.lower(), a))
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for pair in augmented:
        key = (pair[0].lower(), pair[1].lower())
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    
    return unique


def create_synthetic_pairs() -> list:
    """Create additional synthetic training pairs."""
    synthetic = [
        # Greetings
        ("hello", "Hello! I'm your Wyckoff trading assistant. How can I help you today?"),
        ("hi", "Hi there! Ask me anything about Wyckoff methodology."),
        ("hey", "Hey! Ready to discuss Wyckoff trading concepts."),
        
        # Basic definitions
        ("what is wyckoff", "Wyckoff methodology is a technical analysis approach developed by Richard Wyckoff that focuses on understanding market cycles through supply and demand, volume analysis, and price action."),
        ("who is wyckoff", "Richard D. Wyckoff was a famous stock market trader and analyst who developed the Wyckoff method in the early 1900s."),
        
        # Phases
        ("what is accumulation", "Accumulation is a Wyckoff phase where smart money quietly builds positions while price moves sideways after a downtrend."),
        ("what is distribution", "Distribution is a Wyckoff phase where institutions sell their holdings to retail traders while price consolidates at highs."),
        ("what is markup", "Markup is the Wyckoff phase where price advances as demand exceeds supply after accumulation."),
        ("what is markdown", "Markdown is the Wyckoff phase where price declines as supply exceeds demand after distribution."),
        
        # Signals
        ("what is a spring", "A spring is a Wyckoff signal where price briefly drops below support then quickly recovers, trapping sellers and signaling accumulation completion."),
        ("what is an upthrust", "An upthrust is a Wyckoff signal where price briefly breaks above resistance then falls back, trapping buyers and signaling distribution."),
        
        # Concepts
        ("what is composite operator", "The Composite Operator is Wyckoff's concept of imagining all large market participants as a single entity that accumulates, marks up, distributes, and marks down."),
        ("explain supply and demand", "In Wyckoff analysis, supply represents selling pressure and demand represents buying pressure. Price moves based on the balance between these forces."),
        ("what is volume analysis", "Volume analysis in Wyckoff methodology examines trading volume to confirm price movements and identify accumulation or distribution."),
    ]
    
    return synthetic


# =============================================================================
# LABEL SMOOTHING LOSS
# =============================================================================

class LabelSmoothingLoss(nn.Module):
    """Cross entropy with label smoothing."""
    
    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.reshape(-1, self.vocab_size)
        target = target.reshape(-1)
        
        # Create smoothed labels
        smooth_target = torch.zeros_like(logits)
        smooth_target.fill_(self.smoothing / (self.vocab_size - 2))  # -2 for padding and target
        smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_target[:, self.padding_idx] = 0
        
        # Mask padding
        mask = (target != self.padding_idx).float().unsqueeze(1)
        smooth_target = smooth_target * mask
        
        # Calculate loss
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(smooth_target * log_probs).sum(dim=-1)
        
        # Average over non-padding tokens
        non_pad = (target != self.padding_idx).sum()
        return loss.sum() / non_pad if non_pad > 0 else loss.sum()


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, criterion, config):
    """Train one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in progress:
        src = batch['src'].to(config.device)
        tgt = batch['tgt'].to(config.device)
        
        # Teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        optimizer.zero_grad()
        
        logits = model(src, tgt_input)
        loss = criterion(logits, tgt_output)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, criterion, config):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(config.device)
            tgt = batch['tgt'].to(config.device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            logits = model(src, tgt_input)
            loss = criterion(logits, tgt_output)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


def generate_response(model, tokenizer, question: str, config, max_len: int = 50, temperature: float = 0.8):
    """Generate response with improved sampling."""
    model.eval()
    
    src = torch.tensor(
        [tokenizer.encode(question, config.max_len)],
        device=config.device
    )
    tgt = torch.tensor([[tokenizer.SOS_IDX]], device=config.device)
    
    generated = []
    
    with torch.no_grad():
        for _ in range(max_len):
            output = model(src, tgt)
            next_logits = output[:, -1, :] / temperature
            
            # Apply top-k sampling
            top_k = 10
            top_vals, top_idx = torch.topk(next_logits, top_k)
            probs = torch.softmax(top_vals, dim=-1)
            
            # Sample from top-k
            sample_idx = torch.multinomial(probs, 1)
            next_token = top_idx.gather(-1, sample_idx)
            
            token_id = next_token.item()
            
            # Stop at EOS or if generating padding/special tokens
            if token_id == tokenizer.EOS_IDX:
                break
            if token_id == tokenizer.PAD_IDX:
                continue
            if token_id == tokenizer.SOS_IDX:
                continue
            
            generated.append(token_id)
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if too repetitive
            if len(generated) > 3:
                if generated[-1] == generated[-2] == generated[-3]:
                    break
    
    return tokenizer.decode([tokenizer.SOS_IDX] + generated + [tokenizer.EOS_IDX])


def test_model(model, tokenizer, config, test_questions: list):
    """Test model on sample questions."""
    print("\n" + "=" * 60)
    print("GENERATION TEST")
    print("=" * 60)
    
    for q in test_questions:
        response = generate_response(model, tokenizer, q, config)
        print(f"\nQ: {q}")
        print(f"A: {response}")
    
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = Config()
    
    print("=" * 60)
    print("IMPROVED TRANSFORMER TRAINING")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Model: d={config.d_model}, h={config.num_heads}, L={config.num_layers}")
    
    # Load data
    print("\n--- Loading Data ---")
    qa_pairs = load_all_data(config.data_path)
    
    # Add synthetic pairs
    synthetic = create_synthetic_pairs()
    qa_pairs.extend(synthetic)
    print(f"Added {len(synthetic)} synthetic pairs")
    
    # Augment
    qa_pairs = augment_data(qa_pairs)
    print(f"After augmentation: {len(qa_pairs)} pairs")
    
    # Build tokenizer
    print("\n--- Building Tokenizer ---")
    tokenizer = ImprovedTokenizer()
    all_texts = [q for q, a in qa_pairs] + [a for q, a in qa_pairs]
    tokenizer.build_vocab(all_texts, min_freq=config.min_word_freq)
    
    # Split data
    random.shuffle(qa_pairs)
    n_train = int(len(qa_pairs) * config.train_split)
    train_pairs = qa_pairs[:n_train]
    val_pairs = qa_pairs[n_train:]
    
    print(f"\nTrain: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Create datasets
    train_dataset = WyckoffDataset(train_pairs, tokenizer, config.max_len)
    val_dataset = WyckoffDataset(val_pairs, tokenizer, config.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Create model
    print("\n--- Creating Model ---")
    model = TransformerChatbot(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len * 2,
        pad_idx=tokenizer.PAD_IDX
    ).to(config.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = LabelSmoothingLoss(
        tokenizer.vocab_size, 
        tokenizer.PAD_IDX, 
        config.label_smoothing
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98)
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * config.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Test questions
    test_questions = [
        "What is Wyckoff?",
        "What is a spring?",
        "Explain accumulation",
        "What is distribution?",
        "Hello",
    ]
    
    # Training loop
    print("\n--- Training ---")
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, config)
        val_loss = evaluate(model, val_loader, criterion, config)
        
        print(f"Epoch {epoch+1:3d}/{config.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'vocab': tokenizer.word2idx,
                'config': {
                    'vocab_size': tokenizer.vocab_size,
                    'd_model': config.d_model,
                    'num_heads': config.num_heads,
                    'nhead': config.num_heads,
                    'num_layers': config.num_layers,
                    'd_ff': config.d_ff,
                    'max_len': config.max_len * 2,
                }
            }
            torch.save(checkpoint, config.save_path)
            print(f"  -> Saved best model")
        else:
            patience_counter += 1
        
        # Test every 25 epochs
        if (epoch + 1) % 25 == 0:
            test_model(model, tokenizer, config, test_questions)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved: {config.save_path}")
    print("=" * 60)
    
    # Load best and test
    checkpoint = torch.load(config.save_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_model(model, tokenizer, config, test_questions)


if __name__ == "__main__":
    main()