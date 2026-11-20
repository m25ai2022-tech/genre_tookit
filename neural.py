"""Narrative-time neural model (CNN + BiLSTM-attention on GloVe, optional Time2Vec).

Ported from the thesis notebook (temporal_dl_v2_replication.ipynb) so the toolkit uses the same architecture:
  static  : GloVe -> CNN / BiLSTM-attention                       (no temporal information)
  time    : GloVe + Time2Vec(t) fused by addition, with ABEF      (t = i / (n-1) = position inside the paragraph)
The ensemble prediction is the average of the CNN and BiLSTM probabilities.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

EMB_DIM, MAX_LEN = 100, 200
LSTM_UNITS, CONV_FILTERS, CONV_KERNEL = 128, 128, 5
DENSE, DROPOUT = (256, 128), 0.3
D_T, T2V_FMAX = 16, 16.0
DEVICE = "cpu"


def time_vector(n_full, length):
    """t = i/(n-1): position of each token inside its paragraph, in [0, 1]."""
    i = np.arange(length, dtype=np.float32)
    return i / max(n_full - 1, 1)


def to_brown_tokens(tokens):
    """Raw-text quote marks -> the Brown conventions used in training (`` and ''), alternating for straight quotes."""
    out, opened = [], False
    for t in tokens:
        if t in ('"', "\u201c", "\u201d"):
            if t == "\u201c": opened = True
            elif t == "\u201d": opened = False
            else: opened = not opened
            out.append("``" if opened else "''")
        elif t[:1] in ("\u2019", "\u2018"):
            out.append("'" + t[1:])
        elif t in ("\u2014", "\u2013"):
            out.append("--")
        else:
            out.append(t)
    return out


def encode(tokens, stoi):
    """Lower-cased tokens -> padded ids [1, MAX_LEN], time [1, MAX_LEN], length."""
    toks = [t.lower() for t in to_brown_tokens(tokens)]
    ids = [stoi.get(w, 1) for w in toks[:MAX_LEN]]
    L = len(ids)
    I = np.zeros((1, MAX_LEN), np.int64); T = np.zeros((1, MAX_LEN), np.float32)
    I[0, :L] = ids; T[0, :L] = time_vector(len(toks), L)
    return I, T, L


def occurrence_weights(ids, tau, mask, variant):
    """Row i = weights over positions holding the same word as token i (ABEF: uniform; TAEF: recency-weighted)."""
    same = (ids.unsqueeze(2) == ids.unsqueeze(1)) & mask.unsqueeze(1)
    return same.float() / same.float().sum(-1, keepdim=True).clamp_min(1e-9)


class Time2Vec(nn.Module):
    """Time2Vec (Kazemi et al. 2019): [w0*t + b0, sin(wi*t + bi)], all learnable."""
    def __init__(self, dim=D_T, f_max=T2V_FMAX):
        super().__init__()
        k = dim - 1
        self.w0 = nn.Parameter(torch.ones(1)); self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(2 * math.pi * torch.logspace(math.log10(0.5), math.log10(f_max), k))
        self.b = nn.Parameter(torch.rand(k) * 2 * math.pi)

    def forward(self, t):
        t = t.unsqueeze(-1)
        return torch.cat([self.w0 * t + self.b0, torch.sin(t * self.w + self.b)], dim=-1)


class TimeAwareEmbedding(nn.Module):
    """GloVe embedding (fine-tuned) [+ Time2Vec(t) averaged over a word's occurrences (ABEF), fused by addition]."""
    def __init__(self, emb_matrix, time_aware):
        super().__init__()
        self.time_aware = time_aware
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix), freeze=False, padding_idx=0)
        self.out_dim = EMB_DIM
        if time_aware:
            self.t2v = Time2Vec(); self.proj = nn.Linear(D_T, EMB_DIM)

    def forward(self, ids, tau, mask):
        x = self.emb(ids)
        if not self.time_aware:
            return x
        T = torch.bmm(occurrence_weights(ids, tau, mask, "ABEF"), self.t2v(tau))
        return x + self.proj(T)


def make_head(d_in):
    return nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(d_in, DENSE[0]), nn.ReLU(), nn.Dropout(DROPOUT),
                         nn.Linear(DENSE[0], DENSE[1]), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(DENSE[1], 2))


class CNNBranch(nn.Module):
    def __init__(self, temb):
        super().__init__()
        self.temb = temb
        self.conv = nn.Conv1d(temb.out_dim, CONV_FILTERS, CONV_KERNEL, padding=CONV_KERNEL // 2)
        self.head = make_head(CONV_FILTERS)

    def forward(self, ids, tau, mask):
        h = F.relu(self.conv(self.temb(ids, tau, mask).transpose(1, 2)))
        return self.head(h.masked_fill(~mask.unsqueeze(1), -1e4).max(dim=2).values), None


class BiLSTMAttnBranch(nn.Module):
    def __init__(self, temb):
        super().__init__()
        self.temb = temb
        self.lstm = nn.LSTM(temb.out_dim, LSTM_UNITS, batch_first=True, bidirectional=True)
        self.att_W = nn.Linear(2 * LSTM_UNITS, LSTM_UNITS); self.att_v = nn.Linear(LSTM_UNITS, 1, bias=False)
        self.att_drop = nn.Dropout(DROPOUT); self.head = make_head(2 * LSTM_UNITS)

    def forward(self, ids, tau, mask):
        x = self.temb(ids, tau, mask); L = x.size(1)
        out, _ = self.lstm(pack_padded_sequence(x, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False))
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=L)
        a = torch.softmax(self.att_v(torch.tanh(self.att_W(out))).squeeze(-1).masked_fill(~mask, -1e9), dim=-1)
        return self.head(self.att_drop((a.unsqueeze(-1) * out).sum(1))), a


def build(kind, emb, time_aware):
    return (CNNBranch if kind == "cnn" else BiLSTMAttnBranch)(TimeAwareEmbedding(emb, time_aware))


@torch.no_grad()
def predict_one(models, ids, tau, length):
    """models = {'cnn': m, 'bilstm': m}. Returns (P(fiction) of the ensemble, attention over the tokens or None)."""
    ids_t = torch.as_tensor(ids[:, :length]); tau_t = torch.as_tensor(tau[:, :length])
    mask = torch.ones_like(ids_t, dtype=torch.bool)
    probs, att = [], None
    for k, m in models.items():
        m.eval()
        logits, a = m(ids_t, tau_t, mask)
        probs.append(torch.softmax(logits, -1)[0, 1].item())
        if a is not None:
            att = a[0].numpy()
    return float(np.mean(probs)), att
