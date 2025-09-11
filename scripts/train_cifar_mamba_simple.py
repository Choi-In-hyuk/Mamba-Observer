import sys
import os
sys.path.append('/home/choi/choi_ws/mamba')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from einops import rearrange
import time
from tqdm import tqdm

# Import standard Mamba
from mamba_ssm.modules.mamba_simple import Mamba


class SimpleMambaClassifier(nn.Module):
    """Simplified Mamba classifier using mamba_simple.py"""
    def __init__(self, 
                 input_dim=1024,  # 32*32 for CIFAR-10
                 num_classes=10, 
                 d_model=128, 
                 n_layers=4, 
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Project each pixel to d_model (similar to working code)
        self.input_proj = nn.Linear(3, d_model)  # 3 channels for CIFAR-10
        
        # Use regular Mamba layers
        self.mamba_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(n_layers):
            mamba_layer = Mamba(
                d_model=d_model,
                d_state=16,
                d_conv=4,
                expand=2,
                dt_rank="auto",
                layer_idx=i
            )
            self.mamba_layers.append(mamba_layer)
            self.layer_norms.append(nn.LayerNorm(d_model))
        
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
        # x: (B, 3, 32, 32)
        batch_size = x.shape[0]
        
        # Flatten and transpose: (B, 32*32, 3)
        x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)
        
        # Project to d_model
        x = self.input_proj(x)  # (B, 1024, d_model)
        
        # Pass through Mamba layers with residual connections
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = layer_norm(x)
            x = mamba_layer(x)
            x = x + residual
            x = self.dropout(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, d_model)
        
        # Classification
        x = self.final_norm(x)
        return self.classifier(x)


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
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
            'acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(train_loader), correct / total


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader)
    accuracy = correct / total
    return test_loss, accuracy


def train_mamba_simple():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # CIFAR-10 data (keeping color channels)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Dataset & loader
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    # Configuration from working code
    config = {
        'epochs': 15,
        'lr': 1e-3,
        'd_model': 128,  # Increased from working code's 64
        'n_layers': 4,   # Increased from working code's 2
        'weight_decay': 0.01
    }
    
    # Create model
    model = SimpleMambaClassifier(
        input_dim=1024,
        num_classes=10,
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print(f"\n{'='*60}")
    print(f"Training Mamba Simple Model")
    print(f"{'='*60}")
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    best_acc = 0.0
    results = []
    
    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        scheduler.step()
        
        # Testing
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        if test_acc > best_acc:
            best_acc = test_acc
        
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
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%")
        print(f"  Best Acc: {best_acc*100:.2f}%, Time: {epoch_time:.1f}s")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 60)
    
    # Final results
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Final best accuracy: {best_acc*100:.2f}%")
    print(f"Final test accuracy: {test_acc*100:.2f}%")
    
    return results


def quick_test():
    """Quick test to verify the model structure works"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Simple transform for quick test
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Test model
    model = SimpleMambaClassifier(
        d_model=128,
        n_layers=4
    ).to(device)
    
    print("Running quick test...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            print(f"Input shape: {data.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Output sample: {output[0]}")
            print("Quick test successful!")
            break


if __name__ == "__main__":
    # Uncomment one of these:
    train_mamba_simple()    # Full training
    # quick_test()          # Quick test to verify everything works