import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lr_scheduler import NoamScheduler
from typing import Optional
import wandb
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset, collate_fn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.true_dist = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        true_dist = logits.data.clone()
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0
        mask = torch.nonzero(target.data == self.pad_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(logits, true_dist.requires_grad_(False))

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    
    for src, tgt in tqdm(data_iter, desc=f"Epoch {epoch_num}"):
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_idx=0).to(device)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx=0).to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = loss_fn(torch.log_softmax(output, dim=-1).contiguous().view(-1, output.size(-1)), tgt_expected.contiguous().view(-1))
            
            if is_train:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        total_loss += loss.item()

    return total_loss / len(data_iter)

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    
    for i in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx=0).to(device)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        prob = out[:, -1]
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        
        if next_word == end_symbol:
            break
            
    return ys

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    import bleu
    
    model.eval()
    targets = []
    predictions = []
    
    idx_to_word = {v: k for k, v in tgt_vocab.items()}
    
    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            src_mask = make_src_mask(src, pad_idx=0).to(device)
            
            for i in range(src.size(0)):
                single_src = src[i].unsqueeze(0)
                single_src_mask = src_mask[i].unsqueeze(0)
                
                pred_indices = greedy_decode(model, single_src, single_src_mask, max_len, start_symbol=1, end_symbol=2, device=device)
                
                pred_words = [idx_to_word.get(idx.item(), '<unk>') for idx in pred_indices[0] if idx.item() not in [0, 1, 2]]
                target_words = [idx_to_word.get(idx.item(), '<unk>') for idx in tgt[i] if idx.item() not in [0, 1, 2]]
                
                predictions.append(" ".join(pred_words))
                targets.append([" ".join(target_words)])
                
    score = bleu.list_bleu(targets, predictions)
    return score

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    
    model_config = {
        'src_vocab_size': model.src_emb.num_embeddings,
        'tgt_vocab_size': model.tgt_emb.num_embeddings,
        'd_model': model.src_emb.embedding_dim,
        'N': len(model.encoder.layers),
        'num_heads': model.encoder.layers[0].self_attn.num_heads,
        'd_ff': model.encoder.layers[0].ffn.linear1.out_features,
        'dropout': model.encoder.layers[0].self_attn.dropout_layer.p
    }
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'model_config': model_config
    }, path)

def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    return checkpoint['epoch']

def run_training_experiment() -> None:
    wandb.init(project="da6401-a3", config={
        "d_model": 512,
        "n_layers": 6,
        "heads": 8,
        "epochs": 10,
        "batch_size": 32,
        "warmup_steps": 4000
    })
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dataset = Multi30kDataset(split='train')
    val_dataset = Multi30kDataset(split='validation')
    test_dataset = Multi30kDataset(split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
    
    src_vocab_size = len(train_dataset.vocab_de)
    tgt_vocab_size = len(train_dataset.vocab_en)
    
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model=512, N=6, num_heads=8, d_ff=2048).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000)
    loss_fn = LabelSmoothingLoss(tgt_vocab_size, pad_idx=0, smoothing=0.1).to(device)
    
    for epoch in range(10):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, is_train=True, device=device)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, is_train=False, device=device)
        
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch
        })
        
        save_checkpoint(model, optimizer, scheduler, epoch, f"checkpoint_epoch_{epoch}.pt")
        
    bleu = evaluate_bleu(model, test_loader, train_dataset.vocab_en, device)
    wandb.log({'test_bleu': bleu})

if __name__ == "__main__":
    run_training_experiment()
