import argparse
import torch
from torch.utils.data import DataLoader
from train_cnn_spectrogram import SimpleCNN, MelSpecDataset, genSpoof_list, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--protocol", required=True)
parser.add_argument("--base_dir", required=True)
parser.add_argument("--model_path", required=True)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = SimpleCNN().to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
print("Loaded model:", args.model_path)

entries = genSpoof_list(args.protocol)
loader = DataLoader(MelSpecDataset(entries, args.base_dir), batch_size=32, shuffle=False)

eer, acc = evaluate(loader, model, device, tag="eval")
print(f"\nFinal EER: {eer:.2f}%, Accuracy: {acc:.2f}%")