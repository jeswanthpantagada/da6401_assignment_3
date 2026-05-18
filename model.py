import math
import copy
import os
import glob
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Q, K, V shapes:
        [batch, heads, seq_len_q, d_k]
        [batch, heads, seq_len_k, d_k]
        [batch, heads, seq_len_k, d_k]

    mask:
        broadcastable to [batch, heads, seq_len_q, seq_len_k]
        True means masked.
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    attn_weights = torch.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, V)
    return output, attn_weights


def make_src_mask(src: torch.Tensor, pad_idx: int = PAD_IDX) -> torch.Tensor:
    """
    src: [batch, src_len]
    returns: [batch, 1, 1, src_len]
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = PAD_IDX) -> torch.Tensor:
    """
    tgt: [batch, tgt_len]
    returns: [batch, 1, tgt_len, tgt_len]
    """
    batch_size, tgt_len = tgt.size()

    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1,
    )  # [T, T]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, T, T]

    return pad_mask | causal_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        query, key, value:
            [batch, seq_len, d_model]

        returns:
            [batch, seq_len, d_model]
        """
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output, attn_weights = scaled_dot_product_attention(
            Q=Q,
            K=K,
            V=V,
            mask=mask,
            dropout=self.attn_dropout,
        )

        self.last_attn_weights = attn_weights

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(attn_output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_out1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out1))

        attn_out2 = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout2(attn_out2))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        N: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # Store vocabularies for inference
        self.src_vocab: Optional[Dict[str, int]] = None
        self.tgt_vocab: Optional[Dict[str, int]] = None
        self.inv_tgt_vocab: Optional[Dict[int, str]] = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_emb(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def _ensure_vocab(self) -> None:
        """Ensure vocabularies are loaded."""
        if self.src_vocab is not None and self.tgt_vocab is not None and self.inv_tgt_vocab is not None:
            return

        # Try to load from dataset module
        try:
            from dataset import build_vocab_from_train
            self.src_vocab, self.tgt_vocab = build_vocab_from_train()
            self.inv_tgt_vocab = {v: k for k, v in self.tgt_vocab.items()}
        except Exception:
            # Fallback to basic vocab
            if self.src_vocab is None:
                self.src_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
            if self.tgt_vocab is None:
                self.tgt_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
            if self.inv_tgt_vocab is None:
                self.inv_tgt_vocab = {v: k for k, v in self.tgt_vocab.items()}

    @torch.no_grad()
    def infer(self, src_sentence: str, max_len: int = 50) -> str:
        """
        Greedy decoding for a single German sentence.
        Returns the generated English sentence.
        """
        self.eval()
        self._ensure_vocab()

        import spacy

        try:
            spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            try:
                import spacy.cli
                spacy.cli.download("de_core_news_sm")
                spacy_de = spacy.load("de_core_news_sm")
            except:
                spacy_de = spacy.blank("de")

        device = next(self.parameters()).device
        tokens = [tok.text for tok in spacy_de.tokenizer(src_sentence)]
        src_ids = [self.src_vocab.get(tok, UNK_IDX) for tok in tokens]
        src_tensor = torch.tensor([[SOS_IDX] + src_ids + [EOS_IDX]], dtype=torch.long, device=device)

        src_mask = make_src_mask(src_tensor, pad_idx=PAD_IDX).to(device)

        memory = self.encode(src_tensor, src_mask)

        ys = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX).to(device)
            out = self.decode(memory, src_mask, ys, tgt_mask)
            next_word = out[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_word], dim=1)

            if next_word.item() == EOS_IDX:
                break

        words = []
        for idx in ys[0].tolist():
            if idx in (PAD_IDX, SOS_IDX, EOS_IDX):
                continue
            words.append(self.inv_tgt_vocab.get(idx, "<unk>"))

        return " ".join(words)
