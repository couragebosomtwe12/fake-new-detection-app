"""Bidirectional LSTM architecture (Section 3.13): embedding 128, hidden 128
per direction, dropout 0.3.

Defined once and imported by both run_pipeline.py (training) and
app/models_registry.py (inference) so the architecture can never drift
between the two — the same principle as the shared preprocessing module.

Design choices not specified in CLAUDE.md, made here explicitly:
- Vocabulary built from the training split only (min frequency 2, capped at
  20,000 words plus <pad>/<unk>), mirroring the classical TF-IDF vectoriser's
  min_df=2 (see run_pipeline.py's stage_svm/stage_classical).
- No fixed max sequence length ("n/a" per the Section 3.11 preprocessing
  table for classical models) — sequences are packed per-batch instead of
  padded/truncated to a global cap.
"""
from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        num_classes: int = 2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        combined = torch.cat([hidden[0], hidden[1]], dim=1)  # concat both directions
        return self.fc(self.dropout(combined))


def build_vocab(tokenised_texts: list[str], max_features: int = 20000, min_freq: int = 2) -> dict[str, int]:
    """Build a word->id vocabulary from already-preprocessed (lemmatised,
    whitespace-tokenisable) training text. <pad>=0, <unk>=1."""
    counter: Counter[str] = Counter()
    for text in tokenised_texts:
        counter.update(text.split())

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_features + 2:
            break
        vocab[word] = len(vocab)
    return vocab


def encode(preprocessed_text: str, vocab: dict[str, int]) -> list[int]:
    """Map whitespace-split tokens of already-preprocessed text to vocab ids."""
    unk = vocab[UNK_TOKEN]
    ids = [vocab.get(tok, unk) for tok in preprocessed_text.split()]
    return ids or [unk]


def collate_batch(batch: list[tuple[torch.Tensor, int]]):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)
