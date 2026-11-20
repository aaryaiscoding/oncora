```
 ██████╗ ███╗   ██╗ ██████╗ ██████╗ ██████╗  █████╗
██╔═══██╗████╗  ██║██╔════╝██╔═══██╗██╔══██╗██╔══██╗
██║   ██║██╔██╗ ██║██║     ██║   ██║██████╔╝███████║
██║   ██║██║╚██╗██║██║     ██║   ██║██╔══██╗██╔══██║
╚██████╔╝██║ ╚████║╚██████╗╚██████╔╝██║  ██║██║  ██║
 ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
```

# Oncora

Oncora is a clinical decision support project I built around three things: classifying brain tumors from MRI scans, checking prescriptions for safety problems, and combining both into one risk score. The MRI model is trained with federated learning across simulated hospital nodes, so the raw patient images never leave the hospital they belong to.

---

## What it does

**MRI classification.** Upload a brain MRI and the model predicts the tumor type (glioma, meningioma, pituitary, or no tumor) along with a confidence score for each class.

**Prescription safety.** Paste in a prescription and Oncora pulls out the drugs (vocabulary matching first, with a BioBERT NER fallback), checks every pair for interactions, and compares each drug's dose against its typical maximum.

**Fused risk assessment.** A single request takes an MRI and prescription text and merges tumor risk, interaction severity and dose anomaly into one score. It also reports how complete the data was, so anything that couldn't be resolved never gets quietly scored as safe.

**Federated monitoring.** A dashboard showing the global model's test accuracy for each federated round, read from the history file the FL server writes.

---

## Stack

| Layer      | Tech                                                             |
|------------|------------------------------------------------------------------|
| Frontend   | React (Vite)                                                     |
| Backend    | FastAPI                                                          |
| ML model   | ResNet-18 (ImageNet-pretrained, `layer4` + head fine-tuned)      |
| Federation | Flower (`flwr` 1.9), FedProx, 3 simulated hospitals              |
| NLP        | spaCy PhraseMatcher + BioBERT NER, OpenFDA, SQLite caches        |

---

## Federated learning

There are two ways to train the MRI classifier, and the backend loads whichever checkpoint it finds.

| Checkpoint                   | Produced by                  | Notes                                |
|------------------------------|------------------------------|--------------------------------------|
| `models/mri_federated.pt`    | `federated/server.py`        | Backend prefers this one if it exists |
| `models/mri_classifier.pt`   | centralized `train.py`       | Fallback, and the baseline I compare against |

You can force a choice with the `ONCORA_MODEL` environment variable. Both checkpoints use the same head (`Dropout(0.4)` then `Linear`), and the backend still loads older checkpoints that have a plain `Linear` head.

### How a run works

- `federated/split_data.py` splits `datasets/Training` into `datasets/federated/hospital_1` through `hospital_3`.
- Each `federated/client.py` process acts as one hospital. It keeps a fixed 20% of its own images aside for local evaluation, trains on the rest for `LOCAL_EPOCHS` epochs per round, and only sends weights back.
- `federated/server.py` runs **FedProx** with `mu = 0.01`. Aggregation is the same weighted average as FedAvg. The only difference is that each client adds a proximal term to its local loss, which keeps it from drifting too far from the global model on non-IID data.
- After every round the server evaluates the global model on the held-out `datasets/Testing` set, appends the result to `docs/fl_round_history.json`, and overwrites `models/mri_federated.pt`.

Note that the saved checkpoint is the **last** round, not necessarily the best one. The per-round numbers are in `docs/fl_round_history.json`.

### Running it

Start the server first, then one client per hospital, each in its own terminal, all from the `federated/` directory:

```
cd federated
python split_data.py                 # once, after downloading the dataset

python server.py                     # FL_ROUNDS=10 python server.py for a different round count
python client.py 1
python client.py 2
python client.py 3
```

The settings sit at the top of the files. `FL_ROUNDS` (an environment variable, default 5) and `PROXIMAL_MU` are in `server.py`, and `LOCAL_EPOCHS`, `LEARNING_RATE` and `BATCH_SIZE` are in `client.py`. The server listens on port 8080 and waits for 3 clients.

On Colab I run the same commands as background processes with `nohup`, inside a Python 3.11 virtual environment (Flower 1.9 needs `numpy<2`). If port 8080 is already in use on the runtime, switch to another port in both `server.py` and `client.py`. The dataset isn't in the repo, so download the Kaggle Brain Tumor MRI dataset (`Training` and `Testing`) into `datasets/` before you start.

### Results

Global model accuracy on the held-out test set (`datasets/Testing`), FedProx, 3 hospitals. For reference, the untrained model scores about 0.24, which is chance for 4 classes.

| Setup                          | R1     | R2     | R3     | R4     | R5     | Final  |
|--------------------------------|--------|--------|--------|--------|--------|--------|
| 5 rounds, 2 local epochs       | 0.8838 | 0.9150 | 0.9331 | 0.9450 | 0.9431 | 0.9431 |

10 rounds, 5 local epochs:

| R1     | R2     | R3     | R4     | R5     | R6     | R7     | R8     | R9     | R10    |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| 0.9188 | 0.9331 | 0.9394 | 0.9475 | 0.9450 | 0.9450 | 0.9475 | 0.9475 | 0.9488 | 0.9494 |

The 10-round run finished in about 19 minutes and ended at 0.9494 global test accuracy, up from 0.9431 with 5 rounds. Most of the gain came in the first 4 rounds, and after that it only crept up.

`client_accuracy` in the history file is just the clients' own local evaluation, which I keep as a secondary log. It reached 0.9884 in round 10, which is higher than the global test number because the clients evaluate on their own held-out images. The headline number is the server-side `accuracy`.

### FedAvg vs FedProx comparison

`federated/compare_fedavg_fedprox.py` is a separate simulation (Flower + Ray) that runs FedAvg and one or more FedProx values on the same non-IID partitions, then writes a chart and a JSON file to `docs/`. It's reproducible, but it doesn't touch the dashboard history or save a checkpoint.

```
pip install "flwr[simulation]==1.9.0" matplotlib
python -m federated.compare_fedavg_fedprox --num-rounds 5 --mu-values 0.01,0.1 --seed 42
```

Run it from the project root.

---

## Running locally

**One-time setup: prescription NLP pipeline**

```
# 1. Get a free OpenFDA key: https://open.fda.gov/apis/authentication/
cp .env.example .env
# ... fill in OPENFDA_API_KEY in .env

# 2. Install NLP deps (scispaCy's model isn't on PyPI under a normal name,
#    install it via its release URL, per scispaCy's own docs)
pip install -r requirements.txt
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

# 3. Build the drug vocabulary (pages through OpenFDA, takes a while. Run
#    once, then re-run monthly via cron. Do NOT run this at request time.)
python -m backend.nlp.vocab_builder

# 4. Seed the local interaction cache
python -m scripts.init_db
```

**MRI model.** Put `mri_federated.pt` and/or `mri_classifier.pt` in `models/` (`.pt` files are gitignored). You can train the centralized baseline with `train.py`, or use the federated flow above.

**Backend**

```
cd backend
uvicorn main:app --reload
```

The startup log tells you which checkpoint got loaded. BioBERT is optional, so if it fails to load the app just carries on with vocabulary matching.

**Frontend**

```
cd frontend
npm install
npm run dev
```

The frontend falls back to mock data when the backend isn't running, so you can work on the UI on its own.

**Tests**

```
pip install pytest
pytest tests/ -v
```

---

## API

| Method | Endpoint                | Purpose                                                          |
|--------|-------------------------|------------------------------------------------------------------|
| GET    | `/`                     | Status, MRI classes, NLP and BioBERT readiness                   |
| POST   | `/predict-mri`          | MRI image to tumor class, confidence, all probabilities          |
| POST   | `/analyze-prescription` | Drugs, interactions, dose alerts, risk level                     |
| POST   | `/assess-risk`          | MRI + prescription to fused risk score with breakdown            |
| GET    | `/fl-history`           | Per-round federated history (empty list if no run yet)           |
| GET    | `/nlp-sanity-history`   | History of NLP sanity-check rounds                               |
| POST   | `/nlp-sanity-run`       | Run one sanity-check round against the live pipeline             |

`/assess-risk` uses the fusion `0.5 * tumor_risk + 0.3 * interaction_severity + 0.2 * dose_anomaly_score`. It also corrects for two things: raw model confidence isn't the same as risk, and unresolved data must never count as safe. The details are in `backend/risk_assessment.py`.

---

## Evaluation tooling (NLP)

**Accuracy evaluation (fixed hand-labeled sets, run once)**

```
python -m scripts.evaluate_parser
python -m scripts.evaluate_severity
```

**Randomized sanity check (repeatable, tracks accuracy over rounds)**

```
python -m scripts.sanity_check --samples 15
```

This picks random known drug pairs from `data/seed_interactions.csv`, runs them through the live pipeline, and appends the result to `docs/nlp_eval_history.json`. You can see it in the app's **NLP Sanity Check** tab, which tracks rounds the same way the Federated Monitor tab does.

**Dose safety checking.** This works separately from drug-drug interaction checking. It compares a single drug's extracted dose against the typical max single/daily doses in `data/dose_limits.csv`, so "Aspirin 100g" raises a HIGH dose alert even with no other drug in the prescription. Only mass-based units (mg/mcg/g) get compared. ml, IU and units are skipped rather than guessed at. See `backend/nlp/dose_safety.py`.

---

## Project structure

```
oncora/
├── backend/
│   ├── main.py             # FastAPI app and all endpoints
│   ├── risk_assessment.py  # MRI + prescription risk fusion
│   └── nlp/
│       ├── parser.py            # spaCy PhraseMatcher: drug + dosage extraction
│       ├── biobert_ner.py       # optional BioBERT biomedical NER
│       ├── hybrid_ner.py        # combines vocabulary matching and BioBERT
│       ├── vocab_builder.py     # offline job, builds drug_vocab.json from OpenFDA NDC directory
│       ├── openfda_client.py    # OpenFDA HTTP wrapper (auth, retry, backoff)
│       ├── interaction_store.py # SQLite cache (fast path + self-growing write-back)
│       ├── interactions.py      # orchestrates cache -> OpenFDA fallback -> write-back
│       ├── severity.py          # keyword-rule parser: interaction prose -> HIGH/MEDIUM/LOW
│       ├── dose_limit_store.py  # dose limits SQLite store
│       ├── dose_safety.py       # single-drug dose vs. typical safe range checking
│       └── schemas.py           # pydantic request/response models
├── federated/
│   ├── server.py                  # Flower FedProx server: eval, history, checkpoint
│   ├── client.py                  # one simulated hospital
│   ├── split_data.py              # partitions Training into hospital folders
│   └── compare_fedavg_fedprox.py  # FedAvg vs FedProx simulation
├── data/
│   ├── drug_vocab.json         # generated by vocab_builder.py
│   ├── seed_interactions.csv   # curated starting interactions, loaded by scripts/init_db.py
│   ├── dose_limits.csv         # curated max single/daily dose reference table
│   └── interactions.db         # SQLite store (gitignored, grows at runtime)
├── scripts/
│   ├── init_db.py               # one-time/reset DB seeding
│   ├── evaluate_parser.py       # fixed-set drug extraction accuracy (precision/recall/F1)
│   ├── evaluate_severity.py     # fixed-set severity classification accuracy
│   └── sanity_check.py          # randomized repeatable round-based accuracy check
├── tests/                      # pytest suite for the NLP pipeline
├── frontend/                   # React (Vite) dashboard
├── notebooks/                  # dataset loading, centralized training, experimentation
├── models/                     # checkpoints (gitignored)
└── docs/                       # FL history, NLP eval history, charts
```

### Prescription safety pipeline, at a glance

1. `parser.py` pulls out drug names (PhraseMatcher over `drug_vocab.json`) and dosages (regex, scoped to the sentence so each dose lands on the right drug). `hybrid_ner.py` adds any drugs BioBERT finds, when it's available.
2. `interactions.py` checks every detected drug pair against the local SQLite store first.
3. If there's no match, it queries OpenFDA's drug label endpoint and runs the returned text through the keyword rules in `severity.py`.
4. Every result that came from OpenFDA is written back to the store, so coverage grows on its own as new drug pairs show up.
5. If the severity call is low-confidence (ambiguous or unrecognized wording), it gets flagged instead of silently trusted. `InteractionStore.low_confidence_pairs()` gives you a queue to review by hand.

---

## Disclaimer

Oncora is a research project. It is not a medical device and shouldn't be used for clinical decisions.

## Team

Built as part of a federated learning research project.
