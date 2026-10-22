# eval_whisper_backend.py
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor
from train_whisper_backend import WhisperBackendClassifier, SpoofDataset, genSpoof_list, make_collate_fn, evaluate, MODEL_NAME, MAX_DURATION_SEC

parser = argparse.ArgumentParser()
parser.add_argument("--protocol", required=True)
parser.add_argument("--base_dir", required=True)
parser.add_argument("--model_path", required=True)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
model = WhisperBackendClassifier(freeze_encoder=True).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
print("Loaded model:", args.model_path)

entries = genSpoof_list(args.protocol)
max_len_samples = int(MAX_DURATION_SEC * 16000)
loader = DataLoader(SpoofDataset(entries, args.base_dir, feature_extractor, max_len_samples),
                     batch_size=2, shuffle=False, collate_fn=make_collate_fn(feature_extractor))

eer, acc = evaluate(loader, model, device, tag="eval")
print(f"\nFinal EER: {eer:.2f}%, Accuracy: {acc:.2f}%")