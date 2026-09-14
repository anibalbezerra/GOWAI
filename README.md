# GOWAI: Global Observables with Artificial Intelligence for Relativistic Heavy-Ion Collisions

GOWAI is a modular deep learning framework for regressing event-level particle-production observables — charged-particle multiplicity $N_{\rm ch}$ and mean transverse momentum $\langle p_T \rangle$ — from heavy-ion collision initial-condition images. It ships two trained model families (AuAu at 0.2 TeV and PbPb at 2.76 TeV), a unified inference loader, and a single CLI entrypoint (`main.py`) that handles training, evaluation, and batch predictions on new TRENTO events.

---

## Key Features

- **Dual-input, dual-output regressor**: each event is described by
  - a **280×280 dual-channel image** of the initial condition (absolute density + geometry), and
  - a **scalar total-entropy** input, $\sum_i s_i$.
- **Two backbones**, tested on both systems and selected per system:
  - **ResNet-50** (AuAu),
  - **EfficientNetV2-B0** with a learnable 1×1 channel adapter (PbPb).
- **System-specific output heads**, sharing the same overall trunk:
  - **PbPb**: direct regression of normalized $N_{\rm ch}$ and $\langle p_T \rangle$ with a Huber multitask loss.
  - **AuAu**: $N_{\rm ch}$ is regressed directly; $\langle p_T \rangle$ is written as a correction to a fixed, non-trainable degree-2 polynomial baseline, and the loss combines a Huber term on $N_{\rm ch}$ with a multi-quantile pinball loss on $\langle p_T \rangle$ and an evidential Normal–Inverse-Gamma regularizer.
- **Unified loader** (`src/models/HeavyIonRegressor.py`) that resolves the variant, channels, target transform, and normalization automatically from the saved model and its `data_specs.json`.
- **CLI** (`main.py`) supporting prediction (raw files or TFRecord) and legacy training.
- **Pre-trained models** under `src/trained_models/` for the configurations `{energy, entropy} × {all, pt}` for both systems.

---

## Quick Start

### Environment

The repository ships a `pyproject.toml` and `uv.lock`. Create the project virtual environment and install pinned dependencies:

```bash
uv sync
```

Then run commands inside the project's `.venv`:

```bash
uv run python main.py --help
```

### Predict on new events (raw TRENTO files)

```bash
uv run python main.py \
  --system PbPb --energy entropy --all-or-pt pt \
  --x-data my_test_images.txt \
  --predict --save-predictions --plot-results \
  --output-dir ./results
```

### Predict on a TFRecord (inputs already normalized)

```bash
uv run python main.py \
  --system AuAu --energy entropy --all-or-pt pt \
  --tfrecord-file /path/to/test_dataset.tfrecord \
  --predict --save-predictions --plot-results \
  --output-dir ./results
```

If the TFRecord contains labels, the CLI will also denormalize them and save `predictions_from_tfrecord.csv` with columns `Nch_true, Pt_true, Nch_pred, Pt_pred`, plus scatter plots of true vs predicted when `--plot-results` is passed.

---
# DOWNLOAD MODELS

Trained models are hosted on Zenodo: 10.5281/zenodo.22752793. Run python scripts/download_models.py to fetch them into src/trained_models/

---

### Train a new model (legacy path)

```bash
uv run python main.py \
  --system AuAu --energy entropy --all-or-pt pt \
  --x-data training_images.txt --y-data training_labels.txt \
  --train --epochs 50 --batch-size 32 \
  --output-dir ./my_model_results
```

> **Note.** The `--train` path in `main.py` builds a plain ResNet-50 baseline for convenience and does **not** reproduce the shipped AuAu / PbPb models. Production training uses the standalone `regressor_v5.py` (AuAu) and `regressor_v7.py` (PbPb) scripts, which implement the per-system heads, losses, and gap-aware checkpointing.

---

## Data formats

### Text inputs (`--x-data`)

One event per line, 78 400 floats (a flattened 280×280 image). The CLI reshapes each line to `(280, 280)` and builds the dual-channel tensor internally during preprocessing.

### TFRecord inputs (`--tfrecord-file`, `--tfrecord-dir`)

Each record contains:

| Feature | Shape | Content |
|---|---|---|
| `image` | 280×280×2 (bytes) | ch0 = `image / 255`, ch1 = `image / ∑image` |
| `label` | (2,) | `Nch / max_Nch`, `pt / max_pt` (AuAu stores `log(1 + pt/max_pt)`) |
| `sum` | (1,) | `∑ image / max_sum` |

### Predictions output

| File | Columns |
|---|---|
| `predictions.csv` (raw inputs) | `Nch, Pt` |
| `predictions_from_tfrecord.csv` (TFRecord with labels) | `Nch_true, Pt_true, Nch_pred, Pt_pred` |

---

## Model architecture

The regressor is a branched dual-input, dual-output network. Both systems share the same overall structure; they differ only in the backbone, the $\langle p_T \rangle$ head, and the loss.

```
┌──────────────────────────────────────────────────────────────────┐
│                    DUAL-INPUT REGRESSION MODEL                   │
└──────────────────────────────────────────────────────────────────┘

                      ┌─────────────────────────┐
                      │         INPUTS          │
                      │  image (280×280×2)  sum │
                      └─────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
         ┌───────────────┐               ┌────────────────┐
         │ IMAGE BRANCH  │               │ SCALAR BRANCH  │
         │               │               │                │
         │ ResNet-50     │               │ Dense 64 → 32  │
         │   or          │               │       → 16     │
         │ EfficientNet  │               │                │
         │ V2-B0         │               │                │
         │ (+ 1×1 adapter│               │                │
         │  for PbPb)    │               │                │
         │ + CBAM        │               │                │
         └───────────────┘               └────────────────┘
                 │                               │
                 └───────────────┬───────────────┘
                                 │
                        ┌──────────────────┐
                        │  SHARED TRUNK    │
                        │  Concatenate +   │
                        │   Dense stack    │
                        └──────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
         ┌───────────────┐               ┌────────────────┐
         │   Nch HEAD    │               │   pT  HEAD     │
         │   Dense → 1   │               │  see below     │
         └───────────────┘               └────────────────┘
```

### $\langle p_T \rangle$ head — system-specific

**PbPb (v7).** The head is a linear regression on the shared trunk followed by an `AdaptiveBaseline` layer that is a pass-through in the shipped configuration (both polynomial and NN branches disabled), so the output is the normalized $\langle p_T \rangle$ directly.

**AuAu (v5).** The head does not predict $\langle p_T \rangle$ directly. It predicts a single scalar $c$ that corrects a fixed, non-trainable degree-2 polynomial baseline $f(s)$ fitted once on the training set in normalized entropy space:

$$
p_T = c + \log\big(1 + \mathrm{clip}(f(s),\,0,\,1)\big), \qquad
f(s) = a_2 s^2 + a_1 s + a_0,
$$

with $s = \sum_i s_i / \max_{\text{train}} \sum_i s_i$. This encodes the monotonic entropy–$\langle p_T \rangle$ trend as an analytic prior, freeing the network to learn only event-by-event corrections. In addition, an **evidential Normal–Inverse-Gamma head** is attached to the $\langle p_T \rangle$ branch, and its negative log-likelihood plus a regularizer are injected into the training graph via `add_loss`. At inference the head is a no-op: the public output shape stays `(batch, 2)`.

### Losses

- **PbPb (v7).** Huber multitask loss on the two normalized outputs,
  $\mathcal{L} = \mathcal{L}_{N_{\rm ch}}^{\rm Hub} + \mathcal{L}_{\langle p_T \rangle}^{\rm Hub}$.
- **AuAu (v5).** Huber on $N_{\rm ch}$ plus a multi-quantile pinball loss on $\langle p_T \rangle$ at $\tau=\{0.1,0.5,0.9\}$, weighted by $w_{p_T}=4$, plus the evidential NIG term.

### Attention

Both backbones are refined by a spatial attention block (`CBAMBlock` → `SpatialAttention`, 7×7 convolution over channel-wise average and max pooling). This highlights the high-density regions of the initial condition and suppresses noise in the periphery, while keeping the module interpretable through the attention map.

### Why a scalar sum?

The pixel sum of the initial condition is a simple, physics-motivated summary of the event's global energy or entropy content. It correlates strongly with the final multiplicity and is hard for a CNN to recover robustly from pixels at the dataset sizes we use. Providing it explicitly as a second input is what enables the network to achieve the reported accuracy on $N_{\rm ch}$.

---

## Inputs, preprocessing, scaling

### Inputs

- **Images**: 280×280 dual-channel. Channel 0 = absolute density / 255; channel 1 = pixel / ∑pixels (geometry only).
- **Scalar**: total entropy (or energy, depending on the configuration) of the event, normalized by the training-set maximum.

### Preprocessing

For the entropy configurations presented here, TRENTO events are first transformed to entropy density through a lattice-QCD equation of state before being stored. The exact preprocessing is reproduced in `preprocessing_v3.py` (AuAu) and `preprocessing_v4.py` (PbPb).

### Scaling

Each configuration ships a `data_specs.json` with the constants needed for normalization and denormalization:

- `max_sum` — used to normalize the scalar input,
- `max_pixel_raw` (or `x_max`) — used to normalize the image,
- `max_Nch`, `max_pt` — used to normalize the targets,
- `baseline.coefficients` — the fitted degree-2 polynomial (AuAu only).

At inference, the loader reads these automatically from the same directory as the model, so predictions are always returned in physical units.

---

## The unified loader

All the differences between the two model families are hidden behind `HeavyIonRegressor`:

```python
from src.models.HeavyIonRegressor import HeavyIonRegressor

regressor = HeavyIonRegressor("src/trained_models/2.76_PbPb/entropy_pt").load()

# raw events: (N, 280, 280) pixel intensities in physical units
nch, pt = regressor.predict(raw_images, raw_sums)
```

The loader inspects the deserialized model and resolves automatically:

| Aspect | How it is detected |
|---|---|
| Which backbone | from the saved architecture |
| Number of image channels | `model.inputs[0].shape[-1]` |
| Whether `pt_true_input` is required | `"pt_true_input" in model.input_names` |
| Whether $\langle p_T \rangle$ is in log space | presence of an `AddBaseline` layer |
| Target normalization | from `data_specs.json` |
| Polynomial coefficients | from `data_specs.json` |

The CLI (`main.py`) is a thin wrapper around this loader, so the caller never has to branch on the system.

---

## Trained models layout

```
src/trained_models/
├── 0.2_AuAu/
│   ├── energy_all/
│   ├── energy_pt/
│   ├── entropy_all/
│   └── entropy_pt/
│       ├── best_model.architecture.json
│       ├── best_model.weights.h5
│       └── data_specs.json
└── 2.76_PbPb/
    ├── energy_all/
    ├── energy_pt/
    ├── entropy_all/
    └── entropy_pt/
        ├── best_model.architecture.json
        ├── best_model.weights.h5
        └── data_specs.json
```

The loader also accepts a single `best_model.keras` bundle in the same directory; if present, it is preferred over the JSON + H5 pair.

---

## Surrogate model for QGP hydrodynamics

The trained network is a surrogate: it approximates the computationally expensive dynamical simulation (hydrodynamics + hadronization) by learning the mapping from initial-state images (plus the total entropy) directly to event-level observables. When trained on simulation-generated datasets it enables

- fast, differentiable predictions that replicate the full pipeline,
- rapid parameter scans and large-scale inference where full hydrodynamics would be prohibitive,
- cross-system evaluation and studies of initial-condition variability.

As with any surrogate, accuracy depends on the coverage of the training distribution. Keep a validation set and check the predictive uncertainty when applying the model outside its training domain.

---

## Files & structure

```
GOWAI/
├── main.py                          # CLI entrypoint (predict / train)
├── README.md
├── src/
│   ├── models/
│   │   └── HeavyIonRegressor.py     # Unified loader + preprocessor + inference
│   ├── HeavyIonRegressionModel.py   # Legacy model class (baseline training)
│   ├── data_processor.py            # Data ingestion, TFRecord utilities
│   ├── utils/blocks.py              # CBAM / Spatial attention custom layers
│   └── trained_models/              # Pre-trained models (system/observable/selection)
├── preprocessing_v3.py              # AuAu preprocessing (dual-channel + poly baseline)
├── preprocessing_v4.py              # PbPb preprocessing (dual-channel, linear pt)
├── regressor_v5.py                  # AuAu training (evidential head + pinball)
├── regressor_v7.py                  # PbPb training (Huber multitask)
├── run_pipeline.py                  # End-to-end TRENTO → preprocess → inference
├── pyproject.toml
└── uv.lock
```

---

## Next steps

- Unit tests / CI workflows on small synthetic data.
- Cross-system evaluation wrapper.
- Pre-commit hooks.

---

## Citation

If you use this code in published work, please cite our paper:

> Fernando Gardim et al., "GOWAI: A convolutional model for regressing $N_{\rm ch}$ and $\langle p_T \rangle$ from heavy-ion initial conditions", (Add full citation here — to be filled with DOI/venue/volume/pages).