import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import spacy
from torch.nn.utils.rnn import pad_sequence

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

class Multi30kDataset(Dataset):
    def __init__(self, split='train'):
        self.split = split
        self.dataset = load_dataset("bentrevett/multi30k", split=self.split)
        self.vocab_de = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.vocab_en = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.data = []
        
        self.build_vocab()
        self.process_data()

    def build_vocab(self):
        train_data = load_dataset("bentrevett/multi30k", split='train')
        for item in train_data:
            for tok in spacy_de.tokenizer(item['de']):
                if tok.text not in self.vocab_de:
                    self.vocab_de[tok.text] = len(self.vocab_de)
            for tok in spacy_en.tokenizer(item['en']):
                if tok.text not in self.vocab_en:
                    self.vocab_en[tok.text] = len(self.vocab_en)

    def process_data(self):
        for item in self.dataset:
            de_tokens = [self.vocab_de.get(tok.text, 3) for tok in spacy_de.tokenizer(item['de'])]
            en_tokens = [self.vocab_en.get(tok.text, 3) for tok in spacy_en.tokenizer(item['en'])]
            
            de_tensor = torch.tensor([1] + de_tokens + [2])
            en_tensor = torch.tensor([1] + en_tokens + [2])
            
            self.data.append((de_tensor, en_tensor))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    de_batch, en_batch = [], []
    for de_item, en_item in batch:
        de_batch.append(de_item)
        en_batch.append(en_item)
    
    de_batch = pad_sequence(de_batch, padding_value=0, batch_first=True)
    en_batch = pad_sequence(en_batch, padding_value=0, batch_first=True)
    
    return de_batch, en_batch
