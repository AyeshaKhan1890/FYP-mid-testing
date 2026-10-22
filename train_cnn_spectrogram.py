import argparse
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve


def genSpoof_list(protocol_path):
    entries = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            key = parts[1]
            label = parts[-1]
            entries.append((key, 1 if label == "bonafide" else 0))
    return entries


class MelSpecDataset(Dataset):
    def __init__(self, entries, base_dir, n_mels=80, max_frames=200):
        self.entries = entries
        self.base_dir = Path(base_dir)
        self.n_mels = n_mels
        self.max_frames = max_frames

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        key, label = self.entries[idx]
        y, sr = librosa.load(str(self.base_dir / "flac" / f"{key}.flac"), sr=16000)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if mel_db.shape[1] < self.max_frames:
            mel_db = np.pad(mel_db, ((0, 0), (0, self.max_frames - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :self.max_frames]

        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0), label, key


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def compute_eer(bonafide_scores, spoof_scores):
    labels = np.concatenate([np.ones_like(bonafide_scores), np.zeros_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2 * 100


def evaluate(loader, model, device, tag="eval"):
    model.eval()
    all_scores, all_labels = [], []
    total = len(loader)
    start = time.time()
    with torch.no_grad():
        for i, (x, labels, keys) in enumerate(loader):
            x = x.to(device)
            logits = model(x)
            scores = logits[:, 1].cpu().numpy()  # bonafide-class logit as score
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.numpy().tolist())
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"  [{tag}] batch {i+1}/{total} ({time.time()-start:.1f}s)")

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    bonafide = all_scores[all_labels == 1]
    spoof = all_scores[all_labels == 0]
    eer = compute_eer(bonafide, spoof)
    preds = (all_scores > 0).astype(int)
    acc = (preds == all_labels).mean() * 100
    print(f"  bonafide mean={bonafide.mean():.3f}, spoof mean={spoof.mean():.3f}")
    return eer, acc


def train_epoch(loader, model, optimizer, device, epoch_num):
    model.train()
    total_loss = 0
    total = len(loader)
    start = time.time()
    criterion = nn.CrossEntropyLoss()

    for i, (x, labels, keys) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        logits = model(x)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 10 == 0 or (i + 1) == total:
            eta = (time.time()-start)/(i+1)*(total-(i+1))
            print(f"  [epoch {epoch_num}] batch {i+1}/{total} loss={total_loss/(i+1):.5f} (~{eta:.1f}s left)")

    return total_loss / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_protocol", required=True)
    parser.add_argument("--dev_protocol", required=True)
    parser.add_argument("--test_protocol", required=True)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--output_dir", default="./exp_cnn_spectrogram")
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SimpleCNN().to(device)
    nb_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {nb_params:,}")

    train_entries = genSpoof_list(args.train_protocol)
    dev_entries = genSpoof_list(args.dev_protocol)
    test_entries = genSpoof_list(args.test_protocol)
    print(f"Train: {len(train_entries)}, Dev: {len(dev_entries)}, Test: {len(test_entries)}")

    train_loader = DataLoader(MelSpecDataset(train_entries, args.base_dir), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(MelSpecDataset(dev_entries, args.base_dir), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(MelSpecDataset(test_entries, args.base_dir), batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_dev_eer = 100.
    overall_start = time.time()

    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        print(f"\n=== Epoch {epoch}/{args.num_epochs - 1} ===")

        loss = train_epoch(train_loader, model, optimizer, device, epoch)
        dev_eer, dev_acc = evaluate(dev_loader, model, device, tag=f"dev-epoch{epoch}")

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} DONE: loss={loss:.5f}, dev_eer={dev_eer:.2f}%, dev_acc={dev_acc:.2f}%, time={epoch_time:.1f}s")

        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            torch.save(model.state_dict(), output_dir / "best.pth")
            print("  -> new best dev EER, saved checkpoint")

    total_time = time.time() - overall_start
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f} min)")

    print("\n=== Final test evaluation (best dev checkpoint) ===")
    model.load_state_dict(torch.load(output_dir / "best.pth", map_location=device))
    test_eer, test_acc = evaluate(test_loader, model, device, tag="final-test")
    print(f"\nFinal test EER: {test_eer:.2f}%, Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()