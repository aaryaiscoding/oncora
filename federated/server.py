"""
server.py
Flower FL server (FedProx). After every round it:
  1. evaluates the global model on the held-out datasets/Testing set
  2. writes docs/fl_round_history.json (Federated Monitor tab)
  3. writes models/mri_federated.pt (same checkpoint format backend/main.py loads)
"""

import json
import os
from collections import OrderedDict
from datetime import datetime

import flwr as fl
import torch
import torch.nn as nn
from flwr.server.strategy import FedProx
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ---- Paths (absolute, so it works from any working directory) ----
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HISTORY_PATH = os.path.join(ROOT, "docs", "fl_round_history.json")
MODEL_OUT = os.path.join(ROOT, "models", "mri_federated.pt")
TEST_DIR = os.path.join(ROOT, "datasets", "Testing")

# ---- Config ----
NUM_ROUNDS = int(os.environ.get("FL_ROUNDS", 5))
MIN_CLIENTS = 3
PROXIMAL_MU = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must match dataset.py / backend preprocessing
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_data = datasets.ImageFolder(TEST_DIR, transform=eval_tf)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)
CLASS_NAMES = test_data.classes

round_history = []


def build_model():
    m = models.resnet18(weights=None)
    # Must match client.py / train.py head exactly (keys fc.1.weight, fc.1.bias)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, len(CLASS_NAMES)))
    return m.to(DEVICE)


def load_params(model, ndarrays):
    keys = list(model.state_dict().keys())
    if len(keys) != len(ndarrays):
        raise RuntimeError(
            f"Client sent {len(ndarrays)} arrays but ResNet18 state_dict has {len(keys)}. "
            "client.py must send the full state_dict() (weights + BatchNorm buffers)."
        )
    model.load_state_dict(
        OrderedDict((k, torch.tensor(v)) for k, v in zip(keys, ndarrays)), strict=True
    )


def save_history():
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(round_history, f, indent=2)


def save_checkpoint(model, acc, rnd):
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "class_names": CLASS_NAMES,
        "val_accuracy": acc,
        "epoch": rnd,
        "source": "federated-fedprox",
    }, MODEL_OUT)


def evaluate_fn(server_round, parameters, config):
    """Server-side evaluation of the global model on the held-out test set."""
    model = build_model()
    load_params(model, parameters)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss_sum += criterion(out, y).item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    acc, loss = correct / total, loss_sum / total

    if server_round > 0:  # round 0 = untrained initial model, not a real round
        round_history.append({
            "round": server_round,
            "accuracy": acc,
            "loss": loss,
            "timestamp": str(datetime.now()),
        })
        save_history()
        save_checkpoint(model, acc, server_round)
        print(f"\n[Round {server_round}] Global test accuracy: {acc:.4f} | checkpoint + history saved\n")
    return loss, {"accuracy": acc}


def weighted_average(metrics):
    """Client-reported accuracy, kept as a secondary number."""
    total = sum(n for n, _ in metrics)
    avg = sum(n * m["accuracy"] for n, m in metrics) / total
    if round_history:
        round_history[-1]["client_accuracy"] = avg
        save_history()
    print(f"Client-reported (weighted) accuracy: {avg:.4f}")
    return {"accuracy": avg}


strategy = FedProx(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=MIN_CLIENTS,
    min_evaluate_clients=MIN_CLIENTS,
    min_available_clients=MIN_CLIENTS,
    on_fit_config_fn=lambda r: {"server_round": r},
    on_evaluate_config_fn=lambda r: {"server_round": r},
    evaluate_fn=evaluate_fn,
    evaluate_metrics_aggregation_fn=weighted_average,
    proximal_mu=PROXIMAL_MU,
)

if __name__ == "__main__":
    print(f"Starting Flower server (FedProx, mu={PROXIMAL_MU}), {NUM_ROUNDS} rounds, "
          f"waiting for {MIN_CLIENTS} clients. Eval device: {DEVICE}")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
    save_history()
    print("\n--- Federated Learning Complete ---")
    if round_history:
        print(f"Final global test accuracy: {round_history[-1]['accuracy']:.4f}")
    print(f"History: {HISTORY_PATH}\nModel:   {MODEL_OUT}")