# cnn_detector.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# ===================
# 1. Preprocessing: convert galois.FieldArray -> tensor
# ===================
def pubkey_to_vector(pub_key, GF):
    """
    Convert galois.FieldArray public key to integer vector.
    """
    arr = np.array(pub_key, dtype=int)
    return arr.flatten().astype(np.int64)

def preprocess_dataset(keys, GF):
    """
    Normalize keys to [0,1] for neural net input.
    """
    q = GF.order
    X = [pubkey_to_vector(k, GF) for k in keys]
    X = np.stack(X, axis=0)
    X = X.astype(np.float32) / float(q - 1)   # scale to [0,1]
    return X

# ===================
# 2. CNN Model
# ===================
class KeyCNN(nn.Module):
    def __init__(self, input_len, num_classes=2):
        super(KeyCNN, self).__init__()
        # Input is [batch, 1, input_len]
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # Compute conv output size
        conv_out = input_len // 4  # because two poolings halve length twice
        self.fc = nn.Sequential(
            nn.Linear(64 * conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)  # 2 classes: good/bad
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

# ===================
# 3. Training Function
# ===================
def train_cnn(X, y, epochs=10, batch_size=32, lr=1e-3, device="cpu"):
    """
    X: np.array [N, input_len]
    y: np.array [N] labels (0=good, 1=bad)
    """
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    n_val = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_len = X.shape[1]
    model = KeyCNN(input_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb = xb.to(device).unsqueeze(1)  # [B,1,L]
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).unsqueeze(1)
                yb = yb.to(device)
                out = model(xb)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} Loss={avg_loss:.4f} ValAcc={acc:.4f}")

    torch.save(model.state_dict(), "cnn.pth")
    return model

# ===================
# 4. Inference Function
# ===================
def cnn_inference(pub_key, GF, model_path="cnn.pth", device="cpu"):
    """
    Run inference: returns True if suspicious, False if good.
    """
    x = pubkey_to_vector(pub_key, GF).astype(np.float32)
    x = x / float(GF.order - 1)
    x_tensor = torch.tensor(x).unsqueeze(0).unsqueeze(1).to(device)  # [1,1,L]

    input_len = x.shape[0]
    model = KeyCNN(input_len)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        out = model(x_tensor)
        pred = out.argmax(dim=1).item()

    return (pred == 1)  # suspicious if class=1
