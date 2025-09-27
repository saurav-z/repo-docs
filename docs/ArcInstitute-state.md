# State: Predict Cellular Responses with State Transition and Embedding Models

**Predicting cellular responses to perturbations across diverse contexts with State, a powerful framework from the Arc Institute.** ([Original Repository](https://github.com/ArcInstitute/state))

State enables researchers to train state transition (ST) models for predicting cellular responses to perturbations and state embedding (SE) models for analyzing single-cell data.

**Key Features:**

*   **State Transition (ST) Models:** Train models to predict the effects of genetic perturbations.
*   **State Embedding (SE) Models:** Embed and annotate new single-cell datasets.
*   **Flexible Training:** Support for zero-shot and few-shot evaluation paradigms.
*   **Data Preprocessing:** Built-in tools for data normalization, transformation, and HVG selection.
*   **Configuration:** Use TOML files to define datasets, training splits, and evaluation scenarios.
*   **Vector Database Integration:**  Optional integration with LanceDB for efficient similarity search.
*   **Singularity Container:**  Available for consistent and reproducible execution environments.

## Getting Started

Explore the functionality through these examples:

*   Train an ST model for genetic perturbation prediction: [Colab](https://colab.research.google.com/drive/1Ih-KtTEsPqDQnjTh6etVv_f-gRAA86ZN)
*   Perform inference using an ST model trained on Tahoe-100M: [Colab](https://colab.research.google.com/drive/1bq5v7hixnM-tZHwNdgPiuuDo6kuiwLKJ)
*   Embed and annotate a new dataset using SE: [Colab](https://colab.research.google.com/drive/1uJinTJLSesJeot0mP254fQpSxGuDEsZt)
*   Train STATE for the Virtual Cell Challenge: [Colab](https://colab.research.google.com/drive/1QKOtYP7bMpdgDJEipDxaJqOchv7oQ-_l)

## Associated Repositories

*   Model evaluation framework: [cell-eval](https://github.com/ArcInstitute/cell-eval)
*   Dataloaders and preprocessing: [cell-load](https://github.com/ArcInstitute/cell-load)

## Installation

This package is distributed via [`uv`](https://docs.astral.sh/uv).

### Installation from PyPI

```bash
uv tool install arc-state
```

### Installation from Source

```bash
git clone git@github.com:ArcInstitute/state.git
cd state
uv run state
```

For development, install in editable mode:

```bash
git clone git@github.com:ArcInstitute/state.git
cd state
uv tool install -e .
```

## CLI Usage

Access the command-line interface help menu:

```bash
state --help
```

### State Transition Model (ST)

ST models are configured using TOML files.  See `examples/zeroshot.toml` or `examples/fewshot.toml`.

#### Training an ST model:

```bash
state tx train \
data.kwargs.toml_config_path="examples/fewshot.toml" \
data.kwargs.embed_key=X_hvg \
data.kwargs.num_workers=12 \
data.kwargs.batch_col=batch_var \
data.kwargs.pert_col=target_gene \
data.kwargs.cell_type_key=cell_type \
data.kwargs.control_pert=TARGET1 \
training.max_steps=40000 \
training.val_freq=100 \
training.ckpt_every_n_steps=100 \
training.batch_size=8 \
training.lr=1e-4 \
model.kwargs.cell_set_len=64 \
model.kwargs.hidden_dim=328 \
model=pertsets \
wandb.tags="[test]" \
output_dir="$HOME/state" \
name="test"
```

#### Prediction with ST model:

```bash
state tx predict --output-dir $HOME/state/test/ --checkpoint final.ckpt
```

Or, for inference using a trained checkpoint:

```bash
state tx infer --output $HOME/state/test/ --output_dir /path/to/model/ --checkpoint /path/to/model/final.ckpt --adata /path/to/anndata/processed.h5 --pert_col gene --embed_key X_hvg
```

#### Data Preprocessing

**Training Data Preprocessing:**

```bash
state tx preprocess_train \
  --adata /path/to/raw_data.h5ad \
  --output /path/to/preprocessed_training_data.h5ad \
  --num_hvgs 2000
```

**Inference Data Preprocessing:**

```bash
state tx preprocess_infer \
  --adata /path/to/real_data.h5ad \
  --output /path/to/control_template.h5ad \
  --control_condition "DMSO" \
  --pert_col "treatment" \
  --seed 42
```

## TOML Configuration Files

Configure experiments using TOML files, supporting zero-shot and few-shot evaluations.

### Configuration Structure

*   **`[datasets]`**:  Maps dataset names to file paths.
*   **`[training]`**: Specifies datasets for training.
*   **`[zeroshot]`**:  Defines cell types for validation/testing.
*   **`[fewshot]`**: Defines perturbation-level splits within cell types.

### Configuration Examples

*   **Zeroshot Evaluation:**

```toml
[datasets]
replogle = "/data/replogle/"

[training]
replogle = "train"

[zeroshot]
"replogle.jurkat" = "test"
"replogle.rpe1" = "val"

[fewshot]
```

*   **Fewshot Evaluation:**

```toml
[datasets]
replogle = "/data/replogle/"

[training]
replogle = "train"

[zeroshot]

[fewshot]
[fewshot."replogle.k562"]
val = ["AARS"]
test = ["NUP107", "RPUSD4"]

[fewshot."replogle.jurkat"]
val = ["TUFM"]
test = ["MYC", "TP53"]
```

*   **Mixed Evaluation:**

```toml
[datasets]
replogle = "/data/replogle/"

[training]
replogle = "train"

[zeroshot]
"replogle.jurkat" = "test"

[fewshot]
[fewshot."replogle.k562"]
val = ["STAT1"]
test = ["MYC", "TP53"]
```

### Important Notes:

*   Cell types not in `[zeroshot]` and perturbations not in `[fewshot]` are used for training.
*   Perturbations can appear in both validation and test sets in `[fewshot]`.
*   Use the format `"dataset_name.cell_type"` for cell type specifications.
*   Dataset paths should point to directories containing `.h5ad` files.
*   Ensure the `control_pert` condition is available across all splits.

### Validation

The configuration system validates the existence of datasets, cell types, and perturbations.

## State Embedding Model (SE)

### Training SE Model:

```bash
state emb fit --conf ${CONFIG}
```

### Running Inference with SE Model:

```bash
state emb transform \
  --model-folder /large_storage/ctc/userspace/aadduri/SE-600M \
  --checkpoint /large_storage/ctc/userspace/aadduri/SE-600M/se600m_epoch15.ckpt \
  --input /large_storage/ctc/datasets/replogle/rpe1_raw_singlecell_01.h5ad \
  --output /home/aadduri/vci_pretrain/test_output.h5ad
```

### Vector Database (Optional)

Install optional dependencies:

```bash
uv tool install ".[vectordb]"
```

### Build the vector database:

```bash
state emb transform \
  --model-folder /large_storage/ctc/userspace/aadduri/SE-600M \
  --input /large_storage/ctc/public/scBasecamp/GeneFull_Ex50pAS/GeneFull_Ex50pAS/Homo_sapiens/SRX27532045.h5ad \
  --lancedb tmp/state_embeddings.lancedb \
  --gene-column gene_symbols
```

### Query the database:

Obtain embeddings:

```bash
state emb transform \
  --model-folder /large_storage/ctc/userspace/aadduri/SE-600M \
  --input /large_storage/ctc/public/scBasecamp/GeneFull_Ex50pAS/GeneFull_Ex50pAS/Homo_sapiens/SRX27532046.h5ad \
  --output tmp/SRX27532046.h5ad \
  --gene-column gene_symbols
```

Query with embeddings:

```bash
state emb query \
  --lancedb tmp/state_embeddings.lancedb \
  --input tmp/SRX27532046.h5ad \
  --output tmp/similar_cells.csv \
  --k 3
```

## Singularity Container

Build the container:

```bash
singularity build state.sif singularity.def
```

Run the container:

```bash
singularity run state.sif --help
```

Example `state emb transform` run:

```bash
singularity run --nv -B /large_storage:/large_storage \
  state.sif emb transform \
    --model-folder /large_storage/ctc/userspace/aadduri/SE-600M \
    --checkpoint /large_storage/ctc/userspace/aadduri/SE-600M/se600m_epoch15.ckpt \
    --input /large_storage/ctc/datasets/replogle/rpe1_raw_singlecell_01.h5ad \
    --output test_output.h5ad
```

## Licenses

*   Code: [CC BY-NC-SA 4.0](LICENSE)
*   Models/Output: [Arc Research Institute State Model Non-Commercial License](MODEL_LICENSE.md)
*   Acceptable Use: [Arc Research Institute State Model Acceptable Use Policy](MODEL_ACCEPTABLE_USE_POLICY.md)

**Cite:** The State [paper](https://arcinstitute.org/manuscripts/State) when using this code or model parameters.