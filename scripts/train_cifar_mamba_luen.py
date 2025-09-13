import sys
import os


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from einops import rearrange
import time
from tqdm import tqdm

# Import our custom Mamba with Luenberger Observer
from mamba_ssm.modules.mamba_luen import MambaBlockWithObserver


class SimpleMambaClassifierLuenberger(nn.Module):
    """Simplified Mamba classifier with Luenberger Observer"""
    def __init__(self, 
                 input_dim=1024,  # 32*32 for CIFAR-10
                 num_classes=10, 
                 d_model=128, 
                 n_layers=4, 
                 dropout=0.1,
                 observer_alpha=0.3):
        super().__init__()
        self.d_model = d_model
        
        # Project each pixel to d_model (CIFAR-10 has 3 channels)
        self.input_proj = nn.Linear(3, d_model)
        
        # Use MambaBlockWithObserver (Luenberger version)
        self.mamba_layers = MambaBlockWithObserver(
            num_layers=n_layers,
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            use_observer=True,
            observer_alpha=observer_alpha  # Luenberger observer blending weight
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        
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
        # Flatten to sequence: (B, 32*32, 3)
        x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)
        # Project to d_model -> (B, 1024, d_model)
        x = self.input_proj(x)
        
        # Pass through Mamba layers with Luenberger Observer
        x = self.mamba_layers(x)
        
        # Global average pooling over sequence length
        x = x.mean(dim=1)  # (B, d_model)
        
        # Classification head
        x = self.norm(x)
        return self.classifier(x)


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        # Reset Luenberger observer states per batch
        model.mamba_layers.reset_observers()
        
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping for stability
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
            
            # Reset Luenberger observer states for eval
            model.mamba_layers.reset_observers()
            
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader)
    accuracy = correct / total
    return test_loss, accuracy


def save_checkpoint(dir_path, filename, model, optimizer, scheduler, epoch, best_acc, config):
    """Save a training checkpoint inside a directory."""
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


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # CIFAR-10 transforms
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    
    # Datasets & loaders
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Config
    config = {
        'epochs': 100,
        'lr': 2e-3,
        'd_model': 128,
        'n_layers': 4,
        'weight_decay': 0.01,
        'observer_alpha': 0.3  # Luenberger observer blending weight
    }

    # Build model with Luenberger Observer
    model = SimpleMambaClassifierLuenberger(
        input_dim=1024,
        num_classes=10,
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        dropout=0.1,
        observer_alpha=config['observer_alpha']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print(f"\n{'='*60}")
    print(f"Training Mamba with Luenberger Observer (alpha={config['observer_alpha']})")
    print(f"{'='*60}")
    
    optimizer = optim.AdamW(model.parameters(),
                            lr=config['lr'],
                            weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    
    best_acc = 0.0
    results = []
    
    # Checkpoint directory - modified for Luenberger version
    ckpt_dir = "checkpoint_cifar_mamba_luenberger"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best.pth")
    last_path = os.path.join(ckpt_dir, "last.pth")

    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        scheduler.step()
        
        # Eval
        test_loss, test_acc = test(model, test_loader, criterion, device)
        epoch_time = time.time() - start_time
        
        # Save best
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
            print(f"Best checkpoint saved at epoch {epoch} "
                  f"({best_acc*100:.2f}%) -> {best_path}")
        
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
        print(f"  Observer α: {config['observer_alpha']}")
        print("-" * 60)

    # Save last at the end
    save_checkpoint(
        dir_path=ckpt_dir,
        filename="last.pth",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=config['epochs'],
        best_acc=best_acc,
        config=config
    )
    print(f"Last checkpoint saved -> {last_path}")
    
    # Final result
    print("=" * 60)
    print("TRAINING COMPLETED (Luenberger Observer)")
    print("=" * 60)
    print(f"Final best accuracy: {best_acc*100:.2f}%")
    print(f"Observer alpha: {config['observer_alpha']}")
    
    return results


def quick_test():
    """Quick test to verify the Luenberger model structure and forward pass"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    model = SimpleMambaClassifierLuenberger(
        d_model=128,
        n_layers=4,
        observer_alpha=0.3
    ).to(device)
    
    print("Running quick test with Luenberger Observer...")
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            # Reset observer states
            model.mamba_layers.reset_observers()
            output = model(data)
            print(f"Input shape: {data.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Output sample: {output[0]}")
            print("Quick test successful with Luenberger Observer!")
            break


def compare_with_without_observer():
    """Compare performance with and without Luenberger observer"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test with observer
    model_with = SimpleMambaClassifierLuenberger(
        d_model=128, n_layers=4, observer_alpha=0.3
    ).to(device)
    
    # Test without observer  
    model_without = SimpleMambaClassifierLuenberger(
        d_model=128, n_layers=4, observer_alpha=0.0  # No observer correction
    ).to(device)
    
    print(f"Parameters with observer: {sum(p.numel() for p in model_with.parameters()):,}")
    print(f"Parameters without observer: {sum(p.numel() for p in model_without.parameters()):,}")
    
    # Quick forward pass timing
    dummy_input = torch.randn(32, 3, 32, 32).to(device)
    
    with torch.no_grad():
        # With observer
        start = time.time()
        model_with.mamba_layers.reset_observers()
        out_with = model_with(dummy_input)
        time_with = time.time() - start
        
        # Without observer
        start = time.time()
        model_without.mamba_layers.reset_observers()
        out_without = model_without(dummy_input)
        time_without = time.time() - start
        
    print(f"Forward time with observer: {time_with*1000:.2f}ms")
    print(f"Forward time without observer: {time_without*1000:.2f}ms")
    print(f"Time overhead: {((time_with/time_without)-1)*100:.1f}%")


if __name__ == "__main__":
    train_model()    # Train with Luenberger Observer
    # quick_test()
    # compare_with_without_observer()