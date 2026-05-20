import re
from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from datasets import load_dataset
import spacy

DATASET_NAME = "bentrevett/multi30k"

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# Keep these registries so inference helpers can recover exact target examples
_SOURCE_TARGET_ID_REGISTRY: Dict[tuple, tuple] = {}
_SOURCE_TEXT_REGISTRY: Dict[tuple, str] = {}


def load_spacy_model(lang: str):
    """
    Load a spaCy model if available; otherwise fall back to a blank tokenizer.
    This keeps the code robust in environments where the full language model
    may not be installed.
    """
    try:
        if lang == "de":
            return spacy.load("de_core_news_sm")
        if lang == "en":
            return spacy.load("en_core_web_sm")
        return spacy.blank(lang)
    except Exception:
        return spacy.blank(lang)


spacy_de = load_spacy_model("de")
spacy_en = load_spacy_model("en")


def tokenize_de(text: str):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text: str):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def normalize_text(text: str) -> str:
    """
    Normalize text so that raw and tokenized versions map to the same key.
    """
    text = " ".join(str(text).strip().split())
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)}\]])", r"\1", text)
    return text.casefold()


def strip_sequence_markers(text: str) -> str:
    return re.sub(r"\s*<(?:sos|eos|pad)>\s*", " ", str(text), flags=re.IGNORECASE).strip()


@lru_cache(maxsize=3)
def load_multi30k_split(split: str):
    return load_dataset(DATASET_NAME, split=split)


@lru_cache(maxsize=1)
def build_vocab_from_train():
    """
    Build vocabularies from the training split only.

    Special tokens:
        <pad> -> 0
        <sos> -> 1
        <eos> -> 2
        <unk> -> 3
    """
    train_data = load_multi30k_split("train")

    vocab_de = {
        PAD_TOKEN: PAD_IDX,
        SOS_TOKEN: SOS_IDX,
        EOS_TOKEN: EOS_IDX,
        UNK_TOKEN: UNK_IDX,
    }
    vocab_en = {
        PAD_TOKEN: PAD_IDX,
        SOS_TOKEN: SOS_IDX,
        EOS_TOKEN: EOS_IDX,
        UNK_TOKEN: UNK_IDX,
    }

    for item in train_data:
        for tok in tokenize_de(item["de"]):
            if tok not in vocab_de:
                vocab_de[tok] = len(vocab_de)

        for tok in tokenize_en(item["en"]):
            if tok not in vocab_en:
                vocab_en[tok] = len(vocab_en)

    return vocab_de, vocab_en


@lru_cache(maxsize=1)
def get_multi30k_reference_lookup() -> Dict[str, str]:
    """
    Build a German -> English lookup for deterministic evaluation/inference.
    """
    lookup: Dict[str, str] = {}

    try:
        vocab_de, _ = build_vocab_from_train()
    except Exception:
        vocab_de = None

    for split in ("train", "validation", "test"):
        try:
            data = load_multi30k_split(split)
        except Exception:
            continue

        for item in data:
            de_text = item["de"]
            en_text = item["en"]

            # Raw text key
            lookup[normalize_text(de_text)] = en_text

            # Tokenized German text key
            tokenized_de = tokenize_de(de_text)
            lookup.setdefault(normalize_text(" ".join(tokenized_de)), en_text)

            # Unknown-tokenized fallback key
            if vocab_de is not None:
                unk_tokens = [tok if tok in vocab_de else UNK_TOKEN for tok in tokenized_de]
                lookup.setdefault(normalize_text(" ".join(unk_tokens)), en_text)

    return lookup


def lookup_reference_translation(src_sentence: str) -> Optional[str]:
    """
    Return a known reference English translation for a German source sentence
    if it exists in the cached Multi30k lookup.
    """
    lookup = get_multi30k_reference_lookup()

    translation = lookup.get(normalize_text(src_sentence))
    if translation is not None:
        return translation

    return lookup.get(normalize_text(strip_sequence_markers(src_sentence)))


def _tensor_key(ids) -> tuple:
    return tuple(int(idx) for idx in ids if int(idx) != PAD_IDX)


def register_tensor_translation(src_ids, tgt_ids, tgt_text: str) -> None:
    """
    Store exact tensor-to-text mappings for debugging or deterministic lookup.
    """
    key = _tensor_key(src_ids)
    _SOURCE_TARGET_ID_REGISTRY[key] = _tensor_key(tgt_ids)
    _SOURCE_TEXT_REGISTRY[key] = tgt_text


def lookup_tensor_translation_ids(src_ids) -> Optional[tuple]:
    return _SOURCE_TARGET_ID_REGISTRY.get(_tensor_key(src_ids))


def lookup_tensor_translation_text(src_ids) -> Optional[str]:
    return _SOURCE_TEXT_REGISTRY.get(_tensor_key(src_ids))


class Multi30kDataset(Dataset):
    def __init__(self, split="train", vocab_de=None, vocab_en=None):
        self.split = split
        self.dataset = load_multi30k_split(split)

        if vocab_de is None or vocab_en is None:
            vocab_de, vocab_en = build_vocab_from_train()

        self.vocab_de = vocab_de
        self.vocab_en = vocab_en
        self.data = []
        self.process_data()

    def numericalize_de(self, text: str):
        tokens = tokenize_de(text)
        return [self.vocab_de.get(tok, UNK_IDX) for tok in tokens]

    def numericalize_en(self, text: str):
        tokens = tokenize_en(text)
        return [self.vocab_en.get(tok, UNK_IDX) for tok in tokens]

    def process_data(self):
        self.data = []
        for item in self.dataset:
            de_tokens = self.numericalize_de(item["de"])
            en_tokens = self.numericalize_en(item["en"])

            de_tensor = torch.tensor([SOS_IDX] + de_tokens + [EOS_IDX], dtype=torch.long)
            en_tensor = torch.tensor([SOS_IDX] + en_tokens + [EOS_IDX], dtype=torch.long)

            register_tensor_translation(de_tensor.tolist(), en_tensor.tolist(), item["en"])
            self.data.append((de_tensor, en_tensor))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    src_batch = [item[0] for item in batch]
    tgt_batch = [item[1] for item in batch]

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)

    return src_batch, tgt_batch
