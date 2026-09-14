"""
compare_fedavg_fedprox.py

Reproducible FedAvg vs FedProx simulation for Oncora.

The experiment uses the same non-IID hospital partitions and the same
model/training configuration for FedAvg and FedProx. The only algorithmic
difference is the FedProx proximal coefficient.

This version:
- uses CUDA automatically when available;
- accepts an explicit --seed (default 42);
- seeds Python, NumPy, PyTorch, CUDA, and each simulated client;
- uses a deterministic DataLoader generator;
- gives Flower GPU resources when CUDA is available;
- records seed/device information in the JSON output.

The partition itself is controlled separately by partition_data_noniid.py,
which also accepts --seed (default 42).

Run locally:
    python -m federated.compare_fedavg_fedprox \
        --num-rounds 3 --alpha 0.1 --mu-values 0.1,0.5 --seed 42

In Colab, run the same module after cloning/uploading the repository and
installing the project dependencies.
"""

import argparse
import json
import os
import random
from collections import OrderedDict
from pathlib import Path

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg, FedProx
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Prefer deterministic CUDA/cuDNN behavior for reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_generator(seed: int) -> torch.Generator:
    """Create a deterministic PyTorch generator for DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FederatedImageFolder(torch.utils.data.Dataset):
    """
    ImageFolder-like dataset that allows a hospital to have an empty class.

    The global class mapping is fixed:
        glioma     -> 0
        meningioma -> 1
        notumor    -> 2
        pituitary  -> 3
    """

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform

        self.classes = ["glioma", "meningioma", "notumor", "pituitary"]
        self.class_to_idx = {
            "glioma": 0,
            "meningioma": 1,
            "notumor": 2,
            "pituitary": 3,
        }

        self.samples = []

        valid_extensions = {
            ".jpg", ".jpeg", ".png", ".ppm",
            ".bmp", ".pgm", ".tif", ".tiff", ".webp",
        }

        for class_name in self.classes:
            class_dir = self.root / class_name

            if not class_dir.exists():
                continue

            for image_path in class_dir.iterdir():
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in valid_extensions
                ):
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    model = models.resnet18(weights=None)

    for p in model.parameters():
        p.requires_grad = False

    for p in model.layer4.parameters():
        p.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, NUM_CLASSES),
    )

    return model.to(DEVICE)


def get_weights(model):
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def set_weights(model, weights):
    params = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in params}
    )
    model.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------------------------------
# Flower client
# ---------------------------------------------------------------------------
class SimClient(fl.client.NumPyClient):
    """Flower simulation client for one simulated hospital."""

    def __init__(self, hospital_id: int, experiment_seed: int):
        self.hospital_id = hospital_id

        # Give each simulated hospital a deterministic but distinct RNG
        # stream while keeping all strategies reproducible.
        self.client_seed = experiment_seed + hospital_id
        set_seed(self.client_seed)

        self.model = build_model()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        data_dir = DATA_ROOT / f"hospital_{hospital_id}"

        train_data = FederatedImageFolder(
            root=str(data_dir),
            transform=train_transforms,
        )

        self.train_loader = DataLoader(
            train_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=make_generator(self.client_seed),
            num_workers=0,
        )

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        # Re-seed at the beginning of each fit so the same client/round
        # sequence is reproducible across repeated runs.
        server_round = int(config.get("server_round", 0))
        round_seed = self.client_seed + server_round * 1000
        set_seed(round_seed)

        set_weights(self.model, parameters)

        trainable = [
            p for p in self.model.parameters()
            if p.requires_grad
        ]

        global_params = [
            p.clone().detach()
            for p in trainable
        ]

        proximal_mu = float(config.get("proximal_mu", 0.0))

        optimizer = optim.Adam(
            trainable,
            lr=LEARNING_RATE,
            weight_decay=1e-4,
        )

        self.model.train()

        for _ in range(LOCAL_EPOCHS):
            for images, labels in self.train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if proximal_mu > 0:
                    prox_term = sum(
                        (lw - gw).norm(2)
                        for lw, gw in zip(trainable, global_params)
                    )
                    loss = loss + (proximal_mu / 2) * prox_term

                loss.backward()
                optimizer.step()

        return (
            get_weights(self.model),
            len(self.train_loader.dataset),
            {},
        )


# ---------------------------------------------------------------------------
# Centralized global evaluation
# ---------------------------------------------------------------------------
def evaluate_global_model(parameters):
    model = build_model()
    set_weights(model, parameters)

    test_data = datasets.ImageFolder(
        root=str(TEST_ROOT),
        transform=eval_transforms,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_loss = total_loss / total
    test_accuracy = correct / total

    return test_loss, test_accuracy


# ---------------------------------------------------------------------------
# Ray-safe Flower client factory
# ---------------------------------------------------------------------------
def make_client(cid: str, experiment_seed: int):
    """Create a client inside the Ray worker process.

    The function intentionally performs the imports locally so the function
    serialized by Ray does not capture PyTorch/CUDA state from the driver.
    """
    from federated.compare_fedavg_fedprox import SimClient

    hospital_id = int(cid) + 1
    return SimClient(
        hospital_id=hospital_id,
        experiment_seed=experiment_seed,
    ).to_client()


# ---------------------------------------------------------------------------
# Strategy execution
# ---------------------------------------------------------------------------
def run_strategy(
    strategy_name: str,
    num_hospitals: int,
    num_rounds: int,
    proximal_mu: float,
    experiment_seed: int,
):
    # Reset the RNG before each strategy so FedAvg and each FedProx run
    # start from the same reproducible initialization.
    set_seed(experiment_seed)

    round_history = []

    # IMPORTANT: keep the Ray-serialized client function tiny.
    # Do not close over PyTorch/CUDA objects or classes here. Ray serializes
    # this function before sending it to the worker process. The actual
    # SimClient/PyTorch objects are imported and created inside the worker.
    from functools import partial

    client_fn = partial(
        make_client,
        experiment_seed=experiment_seed,
    )

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
            "proximal_mu": proximal_mu,
        },
        evaluate_fn=server_evaluate,
    )

    # Use one deterministic global initialization for every strategy.
    initial_model = build_model()
    initial_parameters = ndarrays_to_parameters(
        get_weights(initial_model)
    )

    if strategy_name == "FedAvg":
        strategy = FedAvg(
            initial_parameters=initial_parameters,
            **common_kwargs,
        )
    else:
        strategy = FedProx(
            proximal_mu=proximal_mu,
            initial_parameters=initial_parameters,
            **common_kwargs,
        )

    if DEVICE.type == "cuda":
        client_resources = {
            "num_cpus": 1,
            "num_gpus": 1.0,
        }
    else:
        client_resources = {
            "num_cpus": 1,
        }

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_hospitals,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    return round_history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sweep(
    fedavg_history,
    fedprox_histories: dict,
    alpha,
    save_path,
):
    """Plot one FedAvg baseline against all tested FedProx values."""

    rounds = range(1, len(fedavg_history) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        rounds,
        fedavg_history,
        marker="o",
        label="FedAvg",
        color="#111111",
        linewidth=2,
    )

    reds = plt.cm.Reds(
        np.linspace(
            0.45,
            0.9,
            max(len(fedprox_histories), 1),
        )
    )

    for color, (mu, history) in zip(
        reds,
        fedprox_histories.items(),
    ):
        ax.plot(
            rounds,
            history,
            marker="o",
            label=f"FedProx (mu={mu})",
            color=color,
        )

    ax.set_xlabel("Federated round")
    ax.set_ylabel("Global model accuracy")
    ax.set_title(
        f"FedAvg vs FedProx under non-IID hospital data "
        f"(alpha={alpha})"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved comparison chart to {save_path}")


def print_summary(fedavg_history, fedprox_histories):
    fedavg_final = fedavg_history[-1]
    fedavg_var = float(np.var(fedavg_history))

    print(
        f"\n{'Strategy':<18}"
        f"{'Final acc':<12}"
        f"{'Round variance':<16}"
    )

    print(
        f"{'FedAvg':<18}"
        f"{fedavg_final:<12.4f}"
        f"{fedavg_var:<16.5f}"
    )

    for mu, history in fedprox_histories.items():
        print(
            f"{'FedProx mu=' + str(mu):<18}"
            f"{history[-1]:<12.4f}"
            f"{float(np.var(history)):<16.5f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-hospitals",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--num-rounds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--mu-values",
        type=str,
        default="0.01",
        help=(
            "Comma-separated FedProx proximal_mu values, "
            "e.g. 0.01,0.1,0.5"
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help=(
            "Chart/log label only. This does NOT regenerate "
            "the partition."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed used for the experiment.",
    )

    args = parser.parse_args()

    experiment_seed = args.seed

    # Set the experiment seed before creating anything stochastic.
    set_seed(experiment_seed)

    mu_values = [
        float(m.strip())
        for m in args.mu_values.split(",")
        if m.strip()
    ]

    print("=" * 60)
    print("Oncora Federated Learning Experiment")
    print(f"Seed: {experiment_seed}")
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"CUDA: {torch.version.cuda}"
        )

    print(f"Alpha label: {args.alpha}")
    print(f"Hospitals: {args.num_hospitals}")
    print(f"Rounds: {args.num_rounds}")
    print(f"FedProx mu values: {mu_values}")
    print("=" * 60)

    print("\nRunning FedAvg (proximal_mu=0.0)...")

    fedavg_history = run_strategy(
        "FedAvg",
        args.num_hospitals,
        args.num_rounds,
        proximal_mu=0.0,
        experiment_seed=experiment_seed,
    )

    fedprox_histories = {}

    for mu in mu_values:
        print(
            f"\nRunning FedProx "
            f"(proximal_mu={mu})..."
        )

        fedprox_histories[mu] = run_strategy(
            "FedProx",
            args.num_hospitals,
            args.num_rounds,
            proximal_mu=mu,
            experiment_seed=experiment_seed,
        )

    plot_sweep(
        fedavg_history,
        fedprox_histories,
        args.alpha,
        DOCS_DIR / f"fedavg_vs_fedprox_alpha{args.alpha}.png",
    )

    print_summary(
        fedavg_history,
        fedprox_histories,
    )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    history_path = (
        DOCS_DIR / "fedavg_vs_fedprox_history.json"
    )

    with open(history_path, "w") as f:
        json.dump(
            {
                "fedavg": fedavg_history,
                "fedprox": {
                    str(mu): history
                    for mu, history in fedprox_histories.items()
                },
                "alpha": args.alpha,
                "seed": experiment_seed,
                "device": str(DEVICE),
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if DEVICE.type == "cuda"
                    else None
                ),
                "config": {
                    "num_hospitals": args.num_hospitals,
                    "num_rounds": args.num_rounds,
                    "batch_size": BATCH_SIZE,
                    "local_epochs": LOCAL_EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "img_size": IMG_SIZE,
                    "num_classes": NUM_CLASSES,
                },
            },
            f,
            indent=2,
        )

    print(f"Saved experiment history to {history_path}")


if __name__ == "__main__":
    main()
