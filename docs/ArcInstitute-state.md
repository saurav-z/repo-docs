# State: Predicting Cellular Responses to Perturbation Across Diverse Contexts

**State enables you to predict cellular responses to perturbations, offering insights into complex biological systems.** For more details, see the original repository at [https://github.com/ArcInstitute/state](https://github.com/ArcInstitute/state).

**Key Features:**

*   **State Transition Model (ST):** Train and utilize models to predict cellular responses to genetic perturbations, including both zero-shot (unseen cell types) and few-shot (limited examples) evaluation.
*   **State Embedding Model (SE):** Embed and annotate new datasets to facilitate deeper analysis and integration.
*   **Data Preprocessing:** Preprocess training and inference data with built-in commands for normalization, log-transformation, and highly variable gene selection.
*   **Flexible Configuration:** Configure experiments using TOML files, supporting dataset specification, training splits, and evaluation scenarios.
*   **Vector Database Integration:** Easily build and query vector databases for efficient similarity searches of embeddings.
*   **Singularity Containerization:** Run the State package in a container for ease of use.

## Getting Started

*   Train an ST model for genetic perturbation prediction using the Replogle-Nadig dataset: [Colab](https://colab.research.google.com/drive/1Ih-KtTEsPqDQnjTh6etVv_f-gRAA86ZN)
*   Perform inference using an ST model trained on Tahoe-100M: [Colab](https://colab.research.google.com/drive/1bq5v7hixnM-tZHwNdgPiuuDo6kuiwLKJ)
*   Embed and annotate a new dataset using SE: [Colab](https://colab.research.google.com/drive/1uJinTJLSesJeot0mP254fQpSxGuDEsZt)
*   Train STATE for the Virtual Cell Challenge: [Colab](https://colab.research.google.com/drive/1QKOtYP7bMpdgDJEipDxaJqOchv7oQ-_l)

## Associated Repositories

*   Model evaluation framework: [cell-eval](https://github.com/ArcInstitute/cell-eval)
*   Dataloaders and preprocessing: [cell-load](https://github.com/ArcInstitute/cell-load)

## Installation

### Installation from PyPI

This package is distributed via [`uv`](https://docs.astral.sh/uv).

```bash
uv tool install arc-state
```

### Installation from Source

```bash
git clone git@github.com:ArcInstitute/state.git
cd state
uv run state
```

When making fundamental changes to State, install an editable version with the `-e` flag.

```bash
git clone git@github.com:ArcInstitute/state.git
cd state
uv tool install -e .
```

## CLI Usage

Access the CLI help menu:

```bash
state --help
```

## State Transition Model (ST)

The ST model predicts cellular responses to perturbations. Experiments are configured with TOML files, specifying datasets and task details.

### Training

Start a new experiment by writing a TOML file (see `examples/zeroshot.toml` or
`examples/fewshot.toml` to start).

Training Example:

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

Ensure that cell lines and perturbations specified in the TOML file match values in  `data.kwargs.cell_type_key` and `data.kwargs.pert_col`.

### Prediction

Use the `tx predict` command to evaluate the ST model:

```bash
state tx predict --output-dir $HOME/state/test/ --checkpoint final.ckpt
```

### Inference

Perform inference on a trained model using the `tx infer` command:

```bash
state tx infer --output $HOME/state/test/ --output_dir /path/to/model/ --checkpoint /path/to/model/final.ckpt --adata /path/to/anndata/processed.h5 --pert_col gene --embed_key X_hvg
```

Where `/path/to/model/` is the folder downloaded from [HuggingFace](https://huggingface.co/arcinstitute).

### Data Preprocessing

#### Training Data Preprocessing

Use `preprocess_train`:

```bash
state tx preprocess_train \
  --adata /path/to/raw_data.h5ad \
  --output /path/to/preprocessed_training_data.h5ad \
  --num_hvgs 2000
```

This command normalizes, log-transforms, and selects highly variable genes, storing the HVG expression matrix in `.obsm['X_hvg']`.

#### Inference Data Preprocessing

Use `preprocess_infer`:

```bash
state tx preprocess_infer \
  --adata /path/to/real_data.h5ad \
  --output /path/to/control_template.h5ad \
  --control_condition "DMSO" \
  --pert_col "treatment" \
  --seed 42
```

Creates a "control template" for inference by replacing perturbed cells with control cell expression.

## TOML Configuration Files

Configure experiments with TOML files to define datasets, training splits, and evaluation scenarios. Supports zeroshot (unseen cell types) and fewshot (limited perturbation examples) evaluation.

### Configuration Structure

*   **`[datasets]`**: Maps dataset names to file system paths.
*   **`[training]`**: Specifies datasets for training.
*   **`[zeroshot]`**: Reserves entire cell types for validation/testing.
*   **`[fewshot]`**: Specifies perturbation-level splits within cell types.

### Configuration Examples

#### Example 1: Pure Zeroshot Evaluation
```toml
# Evaluate generalization to completely unseen cell types
[datasets]
replogle = "/data/replogle/"

[training]
replogle = "train"

[zeroshot]
"replogle.jurkat" = "test"     # Hold out entire jurkat cell line
"replogle.rpe1" = "val"        # Hold out entire rpe1 cell line

[fewshot]
# Empty - no perturbation-level splits
```

#### Example 2: Pure Fewshot Evaluation
```toml
# Evaluate with limited examples of specific perturbations
[datasets]
replogle = "/data/replogle/"

[training]
replogle = "train"

[zeroshot]
# Empty - all cell types participate in training

[fewshot]
[fewshot."replogle.k562"]
val = ["AARS"]                 # Limited AARS examples for validation
test = ["NUP107", "RPUSD4"]    # Limited examples of these genes for testing

[fewshot."replogle.jurkat"]
val = ["TUFM"]
test = ["MYC", "TP53"]
```

#### Example 3: Mixed Evaluation Strategy
```toml
# Combine both zeroshot and fewshot evaluation
[datasets]
replogle = "/data/replogle/"

[training]
replogle = "train"

[zeroshot]
"replogle.jurkat" = "test"        # Zeroshot: unseen cell type

[fewshot]
[fewshot."replogle.k562"]      # Fewshot: limited perturbation examples
val = ["STAT1"]
test = ["MYC", "TP53"]
```

### Important Notes

*   **Automatic training assignment**: Cell types not in `[zeroshot]` automatically train, perturbations not in `[fewshot]` also train
*   **Overlapping splits**: Perturbations can be in both validation and test sets within fewshot
*   **Dataset naming**: Use the format `"dataset_name.cell_type"`
*   **Path requirements**: Dataset paths should point to directories containing h5ad files
*   **Control perturbations**: Ensure control conditions are available across all splits

### Validation

The configuration system validates that:

*   All referenced datasets exist.
*   Cell types in `zeroshot/fewshot` exist in datasets.
*   Perturbations in `fewshot` exist.
*   No conflicts exist between zeroshot and fewshot assignments.

## State Embedding Model (SE)

After installation as above:

```bash
state emb fit --conf ${CONFIG}
```

Run inference with a trained State checkpoint:

```bash
state emb transform \
  --model-folder /large_storage/ctc/userspace/aadduri/SE-600M \
  --checkpoint /large_storage/ctc/userspace/aadduri/SE-600M/se600m_epoch15.ckpt \
  --input /large_storage/ctc/datasets/replogle/rpe1_raw_singlecell_01.h5ad \
  --output /home/aadduri/vci_pretrain/test_output.h5ad
```

Requirements for the h5ad file format:

*   CSR matrix format is required
*   `gene_name` is required in the `var` dataframe

### Vector Database

Install optional dependencies:

```bash
uv tool install ".[vectordb]"
```

Or if having issues:

```bash
uv sync --extra vectordb
```

#### Build the vector database

```bash
state emb transform \
  --model-folder /large_storage/ctc/userspace/aadduri/SE-600M \
  --input /large_storage/ctc/public/scBasecamp/GeneFull_Ex50pAS/GeneFull_Ex50pAS/Homo_sapiens/SRX27532045.h5ad \
  --lancedb tmp/state_embeddings.lancedb \
  --gene-column gene_symbols
```

Running this command multiple times with the same lancedb appends the new data to the provided database.

#### Query the database

Obtain the embeddings:

```bash
state emb transform \
  --model-folder /large_storage/ctc/userspace/aadduri/SE-600M \
  --input /large_storage/ctc/public/scBasecamp/GeneFull_Ex50pAS/GeneFull_Ex50pAS/Homo_sapiens/SRX27532046.h5ad \
  --output tmp/SRX27532046.h5ad \
  --gene-column gene_symbols
```

Query the database:

```bash
state emb query \
  --lancedb tmp/state_embeddings.lancedb \
  --input tmp/SRX27532046.h5ad \
  --output tmp/similar_cells.csv \
  --k 3
```

## Singularity

Containerization is available via `singularity.def`.

Build the container:

```bash
singularity build state.sif singularity.def
```

Run the container:

```bash
singularity run state.sif --help
```

Example of `state emb transform`:

```bash
singularity run --nv -B /large_storage:/large_storage \
  state.sif emb transform \
    --model-folder /large_storage/ctc/userspace/aadduri/SE-600M \
    --checkpoint /large_storage/ctc/userspace/aadduri/SE-600M/se600m_epoch15.ckpt \
    --input /large_storage/ctc/datasets/replogle/rpe1_raw_singlecell_01.h5ad \
    --output test_output.h5ad
```

## Licenses

State code is [licensed](LICENSE) under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0).

Model weights and output are licensed under the [Arc Research Institute State Model Non-Commercial License](MODEL_LICENSE.md) and subject to the [Arc Research Institute State Model Acceptable Use Policy](MODEL_ACCEPTABLE_USE_POLICY.md).

Cite the State [paper](https://arcinstitute.org/manuscripts/State) if you use this code or model parameters.