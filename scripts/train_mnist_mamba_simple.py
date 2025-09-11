import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mamba_ssm.modules.mamba_simple import Mamba  # Mamba block

# Define a simple classifier using Mamba
class MambaClassifier(nn.Module):
    def __init__(self, d_model=128, n_layers=2, num_classes=10):
        super().__init__()
        self.input_proj = nn.Linear(28*28, d_model)
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Flatten MNIST image (B, 1, 28, 28) -> (B, 784)
        x = x.view(x.size(0), -1)
        # Project to model dimension
        x = self.input_proj(x).unsqueeze(1)  # (B, 1, d_model)
        # Pass through Mamba layers
        for layer in self.layers:
            x = layer(x)
        # Take mean over sequence dimension (here length=1)
        x = self.norm(x.mean(dim=1))
        return self.fc(x)

# Training script
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Model
    model = MambaClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, 6):  # 5 epochs
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

        # Evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        print(f"Test Accuracy: {100. * correct / len(test_loader.dataset):.2f}%")

if __name__ == "__main__":
    train()
