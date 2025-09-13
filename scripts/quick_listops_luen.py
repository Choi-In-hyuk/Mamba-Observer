# quickcheck_listops_mamba_luen.py
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Safety toggles (no Intel ext issues)
os.environ['DISABLE_INTEL_EXTENSION_FOR_PYTORCH'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['INTEL_JIT'] = '0'

sys.path.append('/home/choi/choi_ws/mamba')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re
import numpy as np
import random
import time
import argparse

# Use your custom Mamba with Luenberger Observer
from mamba_ssm.modules.mamba_luen import MambaBlockWithObserver


# ------------------------------ Tiny Dataset ------------------------------
class TinyListOpsDataset(Dataset):
    """
    Very small character-level ListOps-like dataset for quick sanity check.
    TSV lines: "<label>\\t<expression>"
    """
    def __init__(self, data, max_length=256, vocab=None):
        self.max_length = max_length
        self.samples = data

        if vocab is None:
            self.vocab = self._build_vocab(self.samples)
        else:
            self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}

    def _build_vocab(self, samples):
        chars = set()
        for line in samples:
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            expr = parts[1].strip()
            for c in expr:
                chars.add(c)
        return ['<PAD>', '<UNK>'] + sorted(list(chars))

    def __len__(self):
        return len(self.samples)

    def _tokenize(self, expr):
        ids = []
        for c in expr:
            ids.append(self.char_to_idx.get(c, self.char_to_idx['<UNK>']))
        return ids

    def __getitem__(self, idx):
        parts = self.samples[idx].split('\t')
        label = int(parts[0])
        expr = parts[1].strip()
        ids = self._tokenize(expr)
        if len(ids) < self.max_length:
            ids = ids + [0] * (self.max_length - len(ids))  # 0 = PAD
        else:
            ids = ids[:self.max_length]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def generate_tiny_listops(num_samples=2000, max_depth=6, max_args=4, seed=0):
    """
    Generate small ListOps-like dataset. Enough to check "learning > 10%".
    """
    random.seed(seed)
    np.random.seed(seed)
    ops = ['MAX', 'MIN', 'MED', 'SM']

    def gen_expr(depth=0):
        if depth >= max_depth:
            return str(random.randint(0, 9))
        nest_p = max(0.8 - depth * 0.2, 0.2)
        if random.random() < nest_p:
            op = random.choice(ops)
            k = random.randint(2, max_args)
            args = []
            for _ in range(k):
                if random.random() < 0.6:
                    args.append(gen_expr(depth + 1))
                else:
                    args.append(str(random.randint(0, 9)))
            return f"[{op} {' '.join(args)}]"
        else:
            return str(random.randint(0, 9))

    def eval_expr(expr):
        try:
            toks = re.findall(r'\[|\]|MAX|MIN|MED|SM|\d+', expr.strip())

            def parse(tokens, p):
                if p >= len(tokens):
                    return 0, p
                if tokens[p] == '[':
                    p += 1
                    if p >= len(tokens): return 0, p
                    op = tokens[p]; p += 1
                    args = []
                    while p < len(tokens) and tokens[p] != ']':
                        if tokens[p] == '[':
                            val, p = parse(tokens, p)
                            args.append(val)
                        else:
                            try:
                                args.append(int(tokens[p]))
                            except:
                                args.append(0)
                            p += 1
                    if p < len(tokens): p += 1  # skip ']'
                    if not args: return 0, p
                    if op == 'MAX': r = max(args)
                    elif op == 'MIN': r = min(args)
                    elif op == 'MED':
                        s = sorted(args); r = s[len(s)//2]
                    elif op == 'SM': r = sum(args) % 10
                    else: r = 0
                    return r, p
                else:
                    try:
                        return int(tokens[p]), p + 1
                    except:
                        return 0, p + 1

            r, _ = parse(toks, 0)
            return r % 10
        except:
            return 0

    data = []
    cnt = 0
    while cnt < num_samples:
        expr = gen_expr(0)
        if '[' not in expr:
            continue
        y = eval_expr(expr)
        data.append(f"{y}\t{expr}")
        cnt += 1
    return data


# ------------------------------ Tiny Model ------------------------------
class TinyMambaClassifier(nn.Module):
    """
    Minimal Mamba classifier with proper PAD handling.
    """
    def __init__(self, vocab_size, num_classes=10, d_model=64, n_layers=1, dropout=0.1, use_observer=True, observer_alpha=0.1, max_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.drop = nn.Dropout(dropout)

        # Sinusoidal PE buffer
        pe = self._make_pe(max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

        self.mamba = MambaBlockWithObserver(
            num_layers=n_layers,
            d_model=d_model,
            d_state=8,
            d_conv=4,
            expand=2,
            use_observer=use_observer,
            observer_alpha=observer_alpha
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

        self.apply(self._init_weights)

    def _make_pe(self, max_len, d):
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0)

    def forward(self, x):
        # x: (B, L)
        mask = (x != 0).float()  # 1 for valid, 0 for PAD
        h = self.embedding(x)
        L = x.size(1)
        h = h + self.pe[:, :L]
        h = h * mask.unsqueeze(-1)  # zero-out PAD even after PE
        h = self.drop(h)

        if hasattr(self.mamba, 'reset_observers'):
            self.mamba.reset_observers()
        h = self.mamba(h)  # (B, L, d)

        # mean pool over valid tokens
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / lengths

        out = self.norm(pooled)
        logits = self.fc(out)
        return logits


# ------------------------------ Train/Eval ------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0*correct/max(1,total):.2f}%")
    return total_loss/len(loader), correct/max(1,total)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss/len(loader), correct/max(1,total)


# ------------------------------ Main ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--observer_alpha", type=float, default=0.1)
    parser.add_argument("--no_observer", action="store_true", help="Disable observer if set")
    args = parser.parse_args()

    # Seeds
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tiny splits: train 2k / val 200 / test 400
    train_data = generate_tiny_listops(2000, max_depth=6, seed=0)
    val_data   = generate_tiny_listops( 200, max_depth=6, seed=1)
    test_data  = generate_tiny_listops( 400, max_depth=6, seed=2)

    # Build datasets with shared vocab
    vocab_probe = TinyListOpsDataset(train_data, max_length=args.max_len)
    train_ds = TinyListOpsDataset(train_data, max_length=args.max_len, vocab=vocab_probe.vocab)
    val_ds   = TinyListOpsDataset(val_data,   max_length=args.max_len, vocab=vocab_probe.vocab)
    test_ds  = TinyListOpsDataset(test_data,  max_length=args.max_len, vocab=vocab_probe.vocab)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size*2, shuffle=False, num_workers=2, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch_size*2, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = TinyMambaClassifier(
        vocab_size=vocab_probe.vocab_size,
        num_classes=10,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=0.1,
        use_observer=(not args.no_observer),
        observer_alpha=args.observer_alpha,
        max_len=args.max_len
    ).to(device)

    # Optim/criterion (no scheduler, no EMA, no label smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Device:", device)
    print("Observer:", not args.no_observer, " Alpha:", args.observer_alpha)

    best_val = 0.0
    t0 = time.time()
    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, optimizer, criterion, device, ep)
        va_loss, va_acc = evaluate(model, val_ld, criterion, device)
        best_val = max(best_val, va_acc)
        print(f"[Epoch {ep}] Train {tr_acc*100:.2f}% | Val {va_acc*100:.2f}% (best {best_val*100:.2f}%)")

    te_loss, te_acc = evaluate(model, test_ld, criterion, device)
    print(f"Test Acc: {te_acc*100:.2f}% | Elapsed: {time.time()-t0:.1f}s")

    # Early sanity threshold print
    if te_acc > 0.15:
        print("OK: Learning surpassed 10-15% sanity threshold.")
    else:
        print("WARN: Still near random (~10%). Check LR/PAD/observer/sizes.")

if __name__ == "__main__":
    main()
