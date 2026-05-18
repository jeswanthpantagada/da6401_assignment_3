import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import spacy

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def load_spacy_model(lang: str):
    try:
        if lang == "de":
            return spacy.load("de_core_news_sm")
        elif lang == "en":
            return spacy.load("en_core_web_sm")
        else:
            return spacy.blank(lang)
    except Exception:
        return spacy.blank(lang)


spacy_de = load_spacy_model("de")
spacy_en = load_spacy_model("en")


def tokenize_de(text: str):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text: str):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def build_vocab_from_train():
    """
    Builds vocabularies from the training split only.
    Special tokens:
        <pad> -> 0
        <sos> -> 1
        <eos> -> 2
        <unk> -> 3
    """
    train_data = load_dataset("bentrevett/multi30k", split="train")

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


class Multi30kDataset(Dataset):
    def __init__(self, split="train", vocab_de=None, vocab_en=None):
        self.split = split
        self.dataset = load_dataset("bentrevett/multi30k", split=split)

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
