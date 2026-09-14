"""
compare_fedavg_fedprox.py
Runs the SAME non-IID hospital partitions (see partition_data_noniid.py)
through two strategies — FedAvg and FedProx — using Flower's simulation
API (virtual clients in-process, no need to manually launch 3 terminals
like client.py/server.py require). Logs per-round accuracy for both and
plots them on one chart.

FedProx reduces to FedAvg exactly when proximal_mu=0.0 (same fact your
server.py docstring already notes), so this script runs the identical
model/data/rounds config twice, varying only mu — that's what makes the
comparison fair and attributable to the algorithm, not a confound.

Requires: flwr, torch, torchvision (same as client.py/server.py).
Requires datasets/federated/hospital_1..N/ to already exist — run
partition_data_noniid.py first.

Run:
    python -m federated.compare_fedavg_fedprox --num-rounds 5 --alpha 0.5 --mu-values 0.01

To actually stress-test FedProx's advantage (mu=0.01 at alpha=0.5 is close
to a no-op — see server.py's own note that mu=0.01 is "nearly mathematically
equivalent to FedAvg"), first repartition with a more extreme skew, then
sweep several mu values against that harder baseline:
    python -m federated.partition_data_noniid --alpha 0.1   # regenerate hospital_*/ dirs
    python -m federated.compare_fedavg_fedprox --num-rounds 5 --alpha 0.1 --mu-values 0.01,0.1,0.5,1.0
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.server.strategy import FedAvg, FedProx
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ---- Config (mirrors federated/client.py exactly, so results are
# comparable to your real client/server run) ----
DEVICE = torch.device("cpu")
IMG_SIZE = 224
BATCH_SIZE = 32
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.0005
NUM_CLASSES = 4
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
DATA_ROOT = _PROJECT_ROOT / "datasets" / "federated"
TEST_ROOT = _PROJECT_ROOT / "datasets" / "Testing"
DOCS_DIR = _PROJECT_ROOT / "docs"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
class FederatedImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform

        self.classes = ["glioma", "meningioma", "notumor", "pituitary"]
        self.class_to_idx = {
            "glioma": 0,
            "meningioma": 1,
            "notumor": 2,
            "pituitary": 3
        }

        self.samples = []

        valid_extensions = {
            ".jpg", ".jpeg", ".png", ".ppm",
            ".bmp", ".pgm", ".tif", ".tiff", ".webp"
        }

        for class_name in self.classes:
            class_dir = self.root / class_name

            if not class_dir.exists():
                continue

            for image_path in class_dir.iterdir():
                if image_path.is_file() and image_path.suffix.lower() in valid_extensions:
                    self.samples.append(
                        (image_path, self.class_to_idx[class_name])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = datasets.folder.default_loader(str(image_path))

        if self.transform:
            image = self.transform(image)

        return image, label
def build_model():
    model = models.resnet18(weights=None)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_features, NUM_CLASSES))
    return model.to(DEVICE)


def get_weights(model):
    return [v.cpu().numpy() for _, v in model.state_dict().items()]


def set_weights(model, weights):
    params = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params})
    model.load_state_dict(state_dict, strict=True)


class SimClient(fl.client.NumPyClient):
    """Same training/eval logic as federated/client.py's HospitalClient,
    adapted to Flower's simulation entrypoint (client_fn takes a string
    cid instead of __init__ taking a hospital_id)."""

    def __init__(self, hospital_id: int):
        self.hospital_id = hospital_id
        self.model = build_model()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        data_dir = DATA_ROOT / f"hospital_{hospital_id}"
        train_data = FederatedImageFolder(root=str(data_dir), transform=train_transforms)
        self.train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        global_params = [p.clone().detach() for p in trainable]
        proximal_mu = config.get("proximal_mu", 0.0)
        optimizer = optim.Adam(trainable, lr=LEARNING_RATE, weight_decay=1e-4)

        self.model.train()
        for _ in range(LOCAL_EPOCHS):
            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                if proximal_mu > 0:
                    prox_term = sum((lw - gw).norm(2) for lw, gw in zip(trainable, global_params))
                    loss = loss + (proximal_mu / 2) * prox_term
                loss.backward()
                optimizer.step()

        return get_weights(self.model), len(self.train_loader.dataset), {}
def evaluate_global_model(parameters):
    model = build_model()
    set_weights(model, parameters)

    test_data = datasets.ImageFolder(
        root=str(TEST_ROOT),
        transform=eval_transforms
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_loss = total_loss / total
    test_accuracy = correct / total

    return test_loss, test_accuracy

def weighted_average(metrics):
    accuracies = [n * m["accuracy"] for n, m in metrics]
    total = sum(n for n, _ in metrics)
    return {"accuracy": sum(accuracies) / total}


def run_strategy(strategy_name: str, num_hospitals: int, num_rounds: int, proximal_mu: float):
    round_history = []

    def client_fn(cid: str):
        return SimClient(hospital_id=int(cid) + 1).to_client()

    # Evaluate the GLOBAL model on the untouched Testing dataset
    def server_evaluate(server_round, parameters, config):
        loss, accuracy = evaluate_global_model(parameters)

        round_history.append(accuracy)

        print(
            f"[{strategy_name}] round {server_round} "
            f"test accuracy: {accuracy:.4f}"
        )

        return loss, {"accuracy": accuracy}

    common_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_hospitals,
        min_available_clients=num_hospitals,
        on_fit_config_fn=lambda server_round: {
            "server_round": server_round,
            "proximal_mu": proximal_mu
        },
        evaluate_fn=server_evaluate,
    )

    if strategy_name == "FedAvg":
        strategy = FedAvg(**common_kwargs)
    else:
        strategy = FedProx(proximal_mu=proximal_mu, **common_kwargs)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_hospitals,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )
    return round_history


def plot_sweep(fedavg_history, fedprox_histories: dict, alpha, save_path):
    """fedprox_histories: {mu_value: round_history} — one FedAvg baseline
    plotted against however many FedProx mu settings were run. Kept as a
    dict (not a single mu) so sweeping --mu-values 0.01,0.1,0.5 plots
    all of them on one chart instead of needing a rerun per value."""
    rounds = range(1, len(fedavg_history) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, fedavg_history, marker="o", label="FedAvg", color="#111111", linewidth=2)

    # Distinct reds so mu values are visually orderable on the chart.
    reds = plt.cm.Reds(np.linspace(0.45, 0.9, max(len(fedprox_histories), 1)))
    for color, (mu, history) in zip(reds, fedprox_histories.items()):
        ax.plot(rounds, history, marker="o", label=f"FedProx (mu={mu})", color=color)

    ax.set_xlabel("Federated round")
    ax.set_ylabel("Global model accuracy")
    ax.set_title(f"FedAvg vs FedProx under non-IID hospital data (alpha={alpha})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved comparison chart to {save_path}")

    # Report both final accuracy AND round-to-round variance, honestly,
    # for every mu tried — no cherry-picking which one goes in the paper.
    # FedProx's proximal term is meant to reduce divergence under client
    # heterogeneity; that can show up as lower variance before it shows
    # up as higher final accuracy, especially in a short run.
    fedavg_final = fedavg_history[-1]
    fedavg_var = float(np.var(fedavg_history))
    print(f"\n{'Strategy':<18}{'Final acc':<12}{'Round variance':<16}")
    print(f"{'FedAvg':<18}{fedavg_final:<12.4f}{fedavg_var:<16.5f}")
    for mu, history in fedprox_histories.items():
        print(f"{'FedProx mu=' + str(mu):<18}{history[-1]:<12.4f}{float(np.var(history)):<16.5f}")
    print("\n(Report whichever setting actually wins — do not pick the mu "
          "value after the fact just because it produced the preferred result.)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-hospitals", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument(
        "--mu-values", type=str, default="0.01",
        help="Comma-separated FedProx proximal_mu values to test against the "
             "same FedAvg baseline, e.g. --mu-values 0.01,0.1,0.5,1.0. Use "
             "higher values (mu=0.01 is nearly identical to plain FedAvg) "
             "combined with a low --alpha to actually stress-test FedAvg's "
             "weakness under heterogeneous client data.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Chart/log label only — this does NOT control partitioning. Run "
             "partition_data_noniid.py with the matching --alpha beforehand "
             "so datasets/federated/hospital_*/ actually reflects it.",
    )
    args = parser.parse_args()
    mu_values = [float(m.strip()) for m in args.mu_values.split(",") if m.strip()]

    print("Running FedAvg (proximal_mu=0.0)...")
    fedavg_history = run_strategy("FedAvg", args.num_hospitals, args.num_rounds, proximal_mu=0.0)






    fedprox_histories = {}
    for mu in mu_values:
        print(f"\nRunning FedProx (proximal_mu={mu})...")
        fedprox_histories[mu] = run_strategy(
            "FedProx", args.num_hospitals, args.num_rounds, proximal_mu=mu
        )

    plot_sweep(
        fedavg_history, fedprox_histories, args.alpha,
        DOCS_DIR / f"fedavg_vs_fedprox_alpha{args.alpha}.png",
    )

    with open(DOCS_DIR / "fedavg_vs_fedprox_history.json", "w") as f:
        json.dump(
            {"fedavg": fedavg_history,
             "fedprox": {str(mu): h for mu, h in fedprox_histories.items()},
             "alpha": args.alpha},
            f, indent=2,
        )


if __name__ == "__main__":
    main()