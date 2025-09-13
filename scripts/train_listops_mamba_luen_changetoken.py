import sys
import os
import warnings
warnings.filterwarnings("ignore")

os.environ['DISABLE_INTEL_EXTENSION_FOR_PYTORCH'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['INTEL_JIT'] = '0'

sys.path.append('/home/choi/choi_ws/mamba')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
from tqdm import tqdm
import re
import numpy as np
from sklearn.metrics import accuracy_score
import pickle

# Import our custom Mamba with Luenberger Observer
from mamba_ssm.modules.mamba_luen import MambaBlockWithObserver


class ImprovedListOpsDataset(Dataset):
    """
    Improved ListOps Dataset with semantic-level tokenization
    """
    def __init__(self, data_path, max_length=256, vocab_size=None):
        self.max_length = max_length
        self.data = []
        self.labels = []
        
        # Build vocabulary at semantic level
        if vocab_size is None:
            self.vocab = self._build_vocab(data_path)
        else:
            self.vocab = vocab_size
            
        self.vocab_size = len(self.vocab)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        
        # Load and preprocess data
        self._load_data(data_path)
        
    def _build_vocab(self, data_path):
        """Build vocabulary with semantic tokens"""
        tokens = set()
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        expression = parts[1]
                        # Extract semantic tokens with regex
                        expr_tokens = re.findall(r'\[|\]|[A-Z]+|\d+', expression)
                        for token in expr_tokens:
                            tokens.add(token)
        
        # Special tokens + semantic tokens
        vocab = ['<PAD>', '<UNK>'] + sorted(list(tokens))
        return vocab
        
    def _load_data(self, data_path):
        """Load data"""
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        label = int(parts[0])
                        expression = parts[1]
                        
                        # Semantic tokenization
                        tokens = self._tokenize(expression)
                        
                        # Length limit
                        if len(tokens) <= self.max_length:
                            self.data.append(tokens)
                            self.labels.append(label)
    
    def _tokenize(self, expression):
        """Semantic tokenization"""
        # Split with regex: [, ], operators (MAX, MIN etc), numbers
        expr_tokens = re.findall(r'\[|\]|[A-Z]+|\d+', expression)
        
        tokens = []
        for token in expr_tokens:
            if token in self.token_to_idx:
                tokens.append(self.token_to_idx[token])
            else:
                tokens.append(self.token_to_idx['<UNK>'])
        return tokens
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        label = self.labels[idx]
        
        # Padding
        if len(tokens) < self.max_length:
            tokens = tokens + [self.token_to_idx['<PAD>']] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
            
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def generate_listops_data(num_samples=2000, max_depth=6, max_args=5, filename=None):
    """
    Generate ListOps data - simpler and more balanced
    """
    import random
    
    operations = ['MAX', 'MIN', 'MED', 'SM']
    
    def generate_expression(depth=0, max_depth=max_depth):
        # Depth limit and simplification
        if depth >= max_depth or random.random() < 0.5:
            return str(random.randint(0, 9))
        
        op = random.choice(operations)
        num_args = random.randint(2, min(4, max_args))  # Limit args count
        args = []
        
        for _ in range(num_args):
            if depth < 2 and random.random() < 0.3:  # Limit nesting
                args.append(generate_expression(depth + 1, max_depth))
            else:
                args.append(str(random.randint(0, 9)))
        
        return f"[{op} {' '.join(args)}]"
    
    def evaluate_expression(expr):
        """Evaluate expression"""
        try:
            tokens = re.findall(r'\[|\]|[A-Z]+|\d+', expr)
            
            def parse_tokens(tokens, pos):
                if pos >= len(tokens):
                    return 0, pos
                    
                if tokens[pos] == '[':
                    pos += 1
                    if pos >= len(tokens):
                        return 0, pos
                        
                    op = tokens[pos]
                    pos += 1
                    args = []
                    
                    while pos < len(tokens) and tokens[pos] != ']':
                        if tokens[pos] == '[':
                            arg, pos = parse_tokens(tokens, pos)
                            args.append(arg)
                        else:
                            try:
                                args.append(int(tokens[pos]))
                            except:
                                args.append(0)
                            pos += 1
                    
                    if pos < len(tokens):
                        pos += 1  # Skip ']'
                    
                    if not args:  # Prevent empty args
                        return 0, pos
                    
                    # Apply operation
                    if op == 'MAX':
                        result = max(args)
                    elif op == 'MIN':
                        result = min(args)
                    elif op == 'MED':
                        result = sorted(args)[len(args) // 2]
                    elif op == 'SM':
                        result = sum(args) % 10
                    else:
                        result = 0
                    
                    return result, pos
                else:
                    try:
                        return int(tokens[pos]), pos + 1
                    except:
                        return 0, pos + 1
            
            result, _ = parse_tokens(tokens, 0)
            return result
        except:
            return 0
    
    # Generate data
    data = []
    for _ in range(num_samples):
        expr = generate_expression()
        label = evaluate_expression(expr)
        data.append(f"{label}\t{expr}")
    
    if filename:
        with open(filename, 'w') as f:
            for line in data:
                f.write(line + '\n')
    
    return data


class ImprovedListOpsMambaClassifier(nn.Module):
    """Improved ListOps Mamba Classifier"""
    def __init__(self, 
                 vocab_size=50,
                 num_classes=10,
                 d_model=256, 
                 n_layers=4,  # Reduced layers
                 dropout=0.1,
                 observer_alpha=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Mamba with Observer
        self.mamba_layers = MambaBlockWithObserver(
            num_layers=n_layers,
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            use_observer=True,
            observer_alpha=observer_alpha
        )
        
        # Improved classification head
        self.norm = nn.LayerNorm(d_model)
        
        # Position weights for last token finding
        self.position_weights = nn.Parameter(torch.ones(1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
        
    def forward(self, x, attention_mask=None):
        # x: (B, L) - token indices
        B, L = x.shape
        
        # Embedding
        x = self.embedding(x)  # (B, L, d_model)
        x = self.embedding_dropout(x)
        
        # Mamba layers
        x = self.mamba_layers(x)
        
        # Improved pooling: use last valid token
        if attention_mask is not None:
            # Use padding mask
            lengths = attention_mask.sum(dim=1) - 1  # Last valid index
            last_hidden = x[torch.arange(B), lengths]
        else:
            # Simply use last token (better method)
            last_hidden = x[:, -1]  # (B, d_model)
        
        # Classification
        x = self.norm(last_hidden)
        return self.classifier(x)


def create_attention_mask(tokens, pad_token_id=0):
    """Create padding mask"""
    return (tokens != pad_token_id).long()


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        # Create padding mask
        attention_mask = create_attention_mask(data, pad_token_id=0)
        attention_mask = attention_mask.to(device)
        
        # Reset observer states
        model.mamba_layers.reset_observers()
        
        optimizer.zero_grad(set_to_none=True)
        output = model(data, attention_mask)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })

    return total_loss / len(train_loader), correct / total


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            attention_mask = create_attention_mask(data, pad_token_id=0)
            attention_mask = attention_mask.to(device)
            
            model.mamba_layers.reset_observers()
            
            output = model(data, attention_mask)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader)
    accuracy = correct / total
    return test_loss, accuracy


def save_checkpoint(dir_path, filename, model, optimizer, scheduler, epoch, best_acc, config):
    """Save checkpoint"""
    os.makedirs(dir_path, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_acc': best_acc,
        'config': config
    }
    torch.save(checkpoint, os.path.join(dir_path, filename))


def prepare_listops_data():
    """Prepare ListOps data"""
    data_dir = "./data/listops"
    os.makedirs(data_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, "train.txt")
    test_path = os.path.join(data_dir, "test.txt")
    
    # Generate new data (overwrite existing)
    print("Generating new ListOps data...")
    generate_listops_data(num_samples=5000, max_depth=4, filename=train_path)
    generate_listops_data(num_samples=1000, max_depth=4, filename=test_path)
    
    return train_path, test_path


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    train_path, test_path = prepare_listops_data()
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = ImprovedListOpsDataset(train_path, max_length=128)  # Shortened length
    test_dataset = ImprovedListOpsDataset(test_path, max_length=128, vocab_size=train_dataset.vocab)
    
    print(f"Vocabulary size: {train_dataset.vocab_size}")
    print(f"Vocabulary sample: {train_dataset.vocab[:20]}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Configuration
    config = {
        'epochs': 30,  # Reduced epochs
        'lr': 5e-4,    # Adjusted learning rate
        'd_model': 256,
        'n_layers': 4,  # Reduced layers
        'weight_decay': 0.01,
        'observer_alpha': 0.05,  # Reduced observer influence
        'vocab_size': train_dataset.vocab_size,
        'num_classes': 10
    }

    # Create model
    model = ImprovedListOpsMambaClassifier(
        vocab_size=config['vocab_size'],
        num_classes=config['num_classes'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        dropout=0.1,
        observer_alpha=config['observer_alpha']
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    print(f"\n{'='*70}")
    print(f"Training Improved ListOps Mamba (Observer alpha={config['observer_alpha']})")
    print(f"{'='*70}")
    
    optimizer = optim.AdamW(model.parameters(),
                            lr=config['lr'],
                            weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    
    best_acc = 0.0
    results = []
    
    # Checkpoint directory
    ckpt_dir = "checkpoint_improved_listops_mamba"
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        scheduler.step()
        
        # Evaluate
        test_loss, test_acc = test(model, test_loader, criterion, device)
        epoch_time = time.time() - start_time
        
        # Save best performance
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(
                dir_path=ckpt_dir,
                filename="best.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_acc=best_acc,
                config=config
            )
        
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc * 100,
            'test_loss': test_loss,
            'test_acc': test_acc * 100,
            'best_acc': best_acc * 100,
            'lr': scheduler.get_last_lr()[0],
            'time': epoch_time
        })
        
        print(f"\nEpoch {epoch}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc*100:.2f}%")
        print(f"  Best Acc: {best_acc*100:.2f}%, Time: {epoch_time:.1f}s")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 70)

    # Final results
    print("=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Final best accuracy: {best_acc*100:.2f}%")
    print(f"Observer alpha: {config['observer_alpha']}")
    print(f"Total parameters: {num_params:,}")
    
    return results


if __name__ == "__main__":
    train_model()