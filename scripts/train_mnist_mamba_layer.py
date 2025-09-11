import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from einops import rearrange

# Import our custom Mamba with Observer
from mamba_ssm.modules.mamba_layer import MambaWithObserver

class PatchEmbedding(nn.Module):
    """Convert image to patch sequence"""
    def __init__(self, img_size=28, patch_size=4, d_model=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 28//4 = 7, so 7*7 = 49 patches
        
        # Patch embedding: conv2d with kernel=patch_size, stride=patch_size
        self.patch_embed = nn.Conv2d(
            in_channels=1, 
            out_channels=d_model, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        
    def forward(self, x):
        # x: (B, 1, 28, 28)
        B = x.shape[0]
        
        # Extract patches: (B, d_model, n_patch_h, n_patch_w)
        x = self.patch_embed(x)  # (B, d_model, 7, 7)
        
        # Flatten spatial dimensions: (B, d_model, n_patches)
        x = rearrange(x, 'b d h w -> b (h w) d')  # (B, 49, d_model)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        return x


class MambaObserverClassifier(nn.Module):
    def __init__(self, d_model=128, n_layers=4, num_classes=10, patch_size=4, use_observer=True):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=28, 
            patch_size=patch_size, 
            d_model=d_model
        )
        
        # Mamba with Observer layers
        self.mamba_layers = MambaWithObserver(
            num_layers=n_layers,
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            use_observer=use_observer,
            observer_d_state=64,
            dropout=0.1
        )
        
        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Global average pooling or use [CLS] token approach
        self.pool_type = "mean"  # "mean" or "cls"
        if self.pool_type == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, x):
        # x: (B, 1, 28, 28)
        B = x.shape[0]
        
        # Convert to patch sequence
        x = self.patch_embed(x)  # (B, n_patches, d_model)
        
        # Add CLS token if using cls pooling
        if self.pool_type == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, d_model)
        
        # Pass through Mamba with Observer
        x = self.mamba_layers(x)  # (B, seq_len, d_model)
        
        # Pooling
        if self.pool_type == "cls":
            x = x[:, 0]  # Take CLS token
        else:
            x = x.mean(dim=1)  # Global average pooling
        
        # Classification
        x = self.norm(x)
        return self.classifier(x)


# Alternative: Pixel-by-pixel processing (more challenging for Observer)
class PixelSequenceClassifier(nn.Module):
    def __init__(self, d_model=64, n_layers=3, num_classes=10, use_observer=True):
        super().__init__()
        
        # Project each pixel to d_model
        self.pixel_embed = nn.Linear(1, d_model)
        
        # Positional embedding for 784 pixels
        self.pos_embed = nn.Parameter(torch.randn(1, 784, d_model))
        
        # Mamba with Observer
        self.mamba_layers = MambaWithObserver(
            num_layers=n_layers,
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            use_observer=use_observer,
            observer_d_state=32,
            dropout=0.1
        )
        
        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x: (B, 1, 28, 28)
        B = x.shape[0]
        
        # Flatten to pixel sequence: (B, 784, 1)
        x = x.view(B, 784, 1)
        
        # Embed each pixel
        x = self.pixel_embed(x)  # (B, 784, d_model)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Pass through Mamba with Observer
        x = self.mamba_layers(x)  # (B, 784, d_model)
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, d_model)
        
        # Classification
        x = self.norm(x)
        return self.classifier(x)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset & loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Model - choose one of the approaches
    print("Training with Patch-based Mamba Observer...")
    model = MambaObserverClassifier(
        d_model=128, 
        n_layers=4, 
        num_classes=10, 
        patch_size=4,
        use_observer=True
    ).to(device)
    
    # Alternative: Pixel-by-pixel approach (uncomment to try)
    # print("Training with Pixel-sequence Mamba Observer...")
    # model = PixelSequenceClassifier(
    #     d_model=64, 
    #     n_layers=3, 
    #     num_classes=10,
    #     use_observer=True
    # ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training loop
    for epoch in range(1, 11):  # 10 epochs
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Reset observer states for each batch
            if hasattr(model, 'mamba_layers'):
                model.mamba_layers.reset_observers()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct_train += pred.eq(target).sum().item()
            total_train += target.size(0)
            
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train
        print(f"Epoch {epoch}")
        print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Evaluation
        model.eval()
        correct = 0
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                # Reset observer states
                if hasattr(model, 'mamba_layers'):
                    model.mamba_layers.reset_observers()
                
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / len(test_loader.dataset)
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 50)


def compare_models():
    """Compare Observer vs No-Observer versions"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Quick test on a small subset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Model with Observer
    model_with_obs = MambaObserverClassifier(use_observer=True).to(device)
    
    # Model without Observer  
    model_without_obs = MambaObserverClassifier(use_observer=False).to(device)
    
    print("Models created for comparison")
    print(f"With Observer params: {sum(p.numel() for p in model_with_obs.parameters()):,}")
    print(f"Without Observer params: {sum(p.numel() for p in model_without_obs.parameters()):,}")


if __name__ == "__main__":
    train()
    # compare_models()  # Uncomment to compare models