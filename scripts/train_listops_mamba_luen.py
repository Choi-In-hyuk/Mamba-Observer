import sys
import os
sys.path.append('/home/choi/choi_ws/mamba')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from tqdm import tqdm
import json
from collections import defaultdict

# Import our custom Mamba with Luenberger Observer
from mamba_ssm.modules.mamba_luen import MambaBlockWithObserver


class ListOpsDataset(Dataset):
    """Dataset for ListOps task"""
    def __init__(self, data_path, vocab=None, max_length=2048):
        self.max_length = max_length
        self.data = []
        self.labels = []
        
        # Build vocab if not provided
        if vocab is None:
            self.vocab = self._build_vocab(data_path)
        else:
            self.vocab = vocab
            
        # Load and process data
        with open(data_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    label = int(parts[0])
                    text = parts[1]
                    
                    # Tokenize
                    tokens = text.replace('(', ' ( ').replace(')', ' ) ').split()
                    token_ids = [self.vocab.get(t, self.vocab['<unk>']) for t in tokens]
                    
                    # Truncate or pad
                    if len(token_ids) > max_length:
                        token_ids = token_ids[:max_length]
                    else:
                        token_ids = token_ids + [self.vocab['<pad>']] * (max_length - len(token_ids))
                    
                    self.data.append(token_ids)
                    self.labels.append(label)
        
        self.data = torch.LongTensor(self.data)
        self.labels = torch.LongTensor(self.labels)
    
    def _build_vocab(self, data_path):
        """Build vocabulary from data"""
        vocab = {'<pad>': 0, '<unk>': 1}
        tokens = set()
        
        with open(data_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    text = parts[1]
                    text_tokens = text.replace('(', ' ( ').replace(')', ' ) ').split()
                    tokens.update(text_tokens)
        
        for idx, token in enumerate(sorted(tokens), start=2):
            vocab[token] = idx
            
        print(f"Vocabulary size: {len(vocab)}")
        return vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MambaListOpsClassifierLuenberger(nn.Module):
    """Mamba classifier with Luenberger Observer for ListOps"""
    def __init__(self,
                 vocab_size,
                 num_classes=10,  # ListOps has outputs 0-9
                 d_model=256,
                 n_layers=6,
                 dropout=0.1,
                 observer_alpha=0.1,
                 max_length=2048):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Positional encoding (learnable)
        self.pos_embedding = nn.Embedding(max_length, d_model)
        
        # Dropout after embedding
        self.embed_dropout = nn.Dropout(dropout)
        
        # Mamba layers with Luenberger Observer
        self.mamba_layers = MambaBlockWithObserver(
            num_layers=n_layers,
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            use_observer=True,
            observer_alpha=observer_alpha
        )
        
        # Output layers
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0, std=0.02)
            if m.padding_idx is not None:
                torch.nn.init.constant_(m.weight[m.padding_idx], 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, mask=None):
        # x: (B, L) token indices
        batch_size, seq_len = x.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens and add positional encoding
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.embed_dropout(x)
        
        # Pass through Mamba layers with Luenberger Observer
        x = self.mamba_layers(x)
        
        # Global average pooling (considering padding if mask provided)
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            x = x.mean(dim=1)
        
        # Classification head
        x = self.norm(x)
        x = self.dropout(x)
        return self.classifier(x)


def create_mask(input_ids, pad_token_id=0):
    """Create attention mask for padded sequences"""
    return (input_ids != pad_token_id).float()


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        mask = create_mask(data)
        
        # Reset Luenberger observer states per batch
        model.mamba_layers.reset_observers()
        
        optimizer.zero_grad(set_to_none=True)
        output = model(data, mask)
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


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            mask = create_mask(data)
            
            # Reset Luenberger observer states for eval
            model.mamba_layers.reset_observers()
            
            output = model(data, mask)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def save_checkpoint(filepath, model, optimizer, scheduler, epoch, best_acc, config):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_acc': best_acc,
        'config': config
    }
    torch.save(checkpoint, filepath)


def train_listops():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    config = {
        'batch_size': 32,
        'epochs': 50,
        'lr': 5e-4,
        'd_model': 256,
        'n_layers': 6,
        'dropout': 0.1,
        'weight_decay': 0.01,
        'observer_alpha': 0.1,  # Luenberger observer weight
        'max_length': 2048,
        'warmup_steps': 1000
    }
    
    # Data paths - adjust these to your ListOps dataset location
    train_path = './data/listops/basic_train.tsv'
    val_path = './data/listops/basic_val.tsv'
    test_path = './data/listops/basic_test.tsv'
    
    print("Loading datasets...")
    
    # Create datasets
    train_dataset = ListOpsDataset(train_path, max_length=config['max_length'])
    val_dataset = ListOpsDataset(val_path, vocab=train_dataset.vocab, max_length=config['max_length'])
    test_dataset = ListOpsDataset(test_path, vocab=train_dataset.vocab, max_length=config['max_length'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'] * 2, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'] * 2, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Build model
    model = MambaListOpsClassifierLuenberger(
        vocab_size=len(train_dataset.vocab),
        num_classes=10,
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        dropout=config['dropout'],
        observer_alpha=config['observer_alpha'],
        max_length=config['max_length']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print(f"\n{'='*60}")
    print(f"Training Mamba with Luenberger Observer on ListOps")
    print(f"Observer alpha: {config['observer_alpha']}")
    print(f"{'='*60}\n")
    
    # Optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < config['warmup_steps']:
            return step / config['warmup_steps']
        return 1.0
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    best_val_acc = 0.0
    results = []
    
    # Checkpoint directory
    ckpt_dir = "checkpoint_listops_mamba_luenberger"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Step scheduler after each epoch
        if epoch > config['warmup_steps'] // len(train_loader):
            scheduler.step()
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                os.path.join(ckpt_dir, 'best.pth'),
                model, optimizer, scheduler, epoch, best_val_acc, config
            )
            print(f"  >> Best model saved (Val Acc: {val_acc*100:.2f}%)")
        
        # Save last model
        save_checkpoint(
            os.path.join(ckpt_dir, 'last.pth'),
            model, optimizer, scheduler, epoch, best_val_acc, config
        )
        
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc * 100,
            'val_loss': val_loss,
            'val_acc': val_acc * 100,
            'best_val_acc': best_val_acc * 100,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time
        })
        
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        print(f"  Best Val Acc: {best_val_acc*100:.2f}%")
        print(f"  Time: {epoch_time:.1f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 60)
    
    # Final test evaluation
    print("\n" + "="*60)
    print("Final Test Evaluation")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(os.path.join(ckpt_dir, 'best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"Observer Alpha: {config['observer_alpha']}")
    
    # Save results
    with open(os.path.join(ckpt_dir, 'results.json'), 'w') as f:
        json.dump({
            'config': config,
            'results': results,
            'test_acc': test_acc * 100,
            'best_val_acc': best_val_acc * 100
        }, f, indent=2)
    
    return results


def compare_observer_ablation():
    """Compare different observer alpha values"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test different alpha values
    alphas = [0.0, 0.05, 0.1, 0.15, 0.2]
    
    print("="*60)
    print("Observer Alpha Ablation Study")
    print("="*60)
    
    # Create a small test dataset for quick comparison
    train_path = './data/listops/basic_train.tsv'
    dataset = ListOpsDataset(train_path, max_length=512)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    vocab_size = len(dataset.vocab)
    
    for alpha in alphas:
        model = MambaListOpsClassifierLuenberger(
            vocab_size=vocab_size,
            d_model=128,
            n_layers=4,
            observer_alpha=alpha
        ).to(device)
        
        # Quick forward pass test
        model.eval()
        times = []
        
        with torch.no_grad():
            for _ in range(10):  # Average over 10 runs
                data, _ = next(iter(loader))
                data = data.to(device)
                
                start = time.time()
                model.mamba_layers.reset_observers()
                _ = model(data)
                times.append(time.time() - start)
        
        avg_time = np.mean(times[2:])  # Skip first 2 for warmup
        print(f"Alpha={alpha:.2f}: {avg_time*1000:.2f}ms per batch")
    
    print("="*60)


def download_listops_data():
    """Helper function to download ListOps dataset"""
    import urllib.request
    import zipfile
    
    print("Downloading ListOps dataset...")
    
    # Create data directory
    os.makedirs('./data/listops', exist_ok=True)
    
    # URLs for ListOps dataset (you may need to adjust these)
    base_url = "https://github.com/nyu-mll/GLUE-baselines/raw/master/data/ListOps/"
    files = ['basic_train.tsv', 'basic_val.tsv', 'basic_test.tsv']
    
    for file in files:
        filepath = f'./data/listops/{file}'
        if not os.path.exists(filepath):
            print(f"Downloading {file}...")
            try:
                urllib.request.urlretrieve(base_url + file, filepath)
                print(f"  {file} downloaded successfully")
            except:
                print(f"  Could not download {file} from {base_url}")
                print("  Please download the ListOps dataset manually")
                return False
    
    print("Dataset download complete!")
    return True


if __name__ == "__main__":
    # Check if data exists, if not try to download
    if not os.path.exists('./data/listops/basic_train.tsv'):
        print("ListOps dataset not found.")
        if not download_listops_data():
            print("\nPlease download the ListOps dataset manually and place it in ./data/listops/")
            print("Required files: basic_train.tsv, basic_val.tsv, basic_test.tsv")
            exit(1)
    
    # Train the model
    train_listops()
    
    # Optional: Run ablation study
    # compare_observer_ablation()