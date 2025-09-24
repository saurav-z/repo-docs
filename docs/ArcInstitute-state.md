# State: Predict Cellular Responses to Perturbation with Deep Learning

**Uncover the future of cellular biology: leverage State, a powerful deep learning framework, to predict how cells react to diverse perturbations across varied biological contexts.**  [View the original repository on GitHub](https://github.com/ArcInstitute/state)

State empowers researchers to train state transition models (ST) and pretrain state embedding models (SE) for analyzing single-cell RNA sequencing (scRNA-seq) data.  See the [State paper](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2) for details.

**Key Features:**

*   **State Transition (ST) Models:** Predict cellular responses to genetic perturbations, enabling zero-shot and few-shot evaluation paradigms.
*   **State Embedding (SE) Models:**  Generate powerful embeddings for scRNA-seq data to facilitate data analysis and integration.
*   **Flexible Configuration:** Utilize TOML configuration files to define datasets, training strategies (zeroshot, fewshot), and evaluation protocols.
*   **Data Preprocessing:**  Provides CLI tools for training and inference data preprocessing, including normalization, log-transformation, and highly variable gene selection.
*   **Vector Database Integration:**  Integrates with LanceDB for efficient similarity search and analysis of cell embeddings.
*   **Containerization:** Supports deployment via Singularity containers for reproducibility and ease of use.

**Associated Repositories:**

*   Model evaluation framework: [cell-eval](https://github.com/ArcInstitute/cell-eval)
*   Dataloaders and preprocessing: [cell-load](https://github.com/ArcInstitute/cell-load)

## Getting Started

Explore State's capabilities with these interactive tutorials:

*   Train an ST model for genetic perturbation prediction: [Colab](https://colab.research.google.com/drive/1Ih-KtTEsPqDQnjTh6etVv_f-gRAA86ZN)
*   Perform inference with an ST model: [Colab](https://colab.research.google.com/drive/1bq5v7hixnM-tZHwNdgPiuuDo6kuiwLKJ)
*   Embed and annotate a new dataset using SE: [Colab](https://colab.research.google.com/drive/1uJinTJLSesJeot0mP254fQpSxGuDEsZt)
*   Train STATE for the Virtual Cell Challenge: [Colab](https://colab.research.google.com/drive/1QKOtYP7bMpdgDJEipDxaJqOchv7oQ-_l)

## Installation

State is easily installable using `uv` or from source.

### Installation with `uv`

```bash
uv tool install arc-state
```

### Installation from Source

```bash
git clone git@github.com:ArcInstitute/state.git
cd state
uv run state
```

Install an editable version for development:

```bash
git clone git@github.com:ArcInstitute/state.git
cd state
uv tool install -e .
```

## CLI Usage

Access the command-line interface (CLI) for model training, inference, and data preprocessing.

Get help on the CLI:

```bash
state --help
```

## State Transition Model (ST) - Predicting Perturbation Effects

ST models predict the effects of perturbations on single-cell data.

1.  **Configuration:** Start by creating a TOML configuration file (e.g., `examples/fewshot.toml`) to define datasets, training splits, and evaluation scenarios.
2.  **Training:** Train an ST model using the `state tx train` command:

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

3.  **Prediction/Inference:** Use `state tx predict` to evaluate a trained model or `state tx infer` for inference on new data:

    ```bash
    state tx predict --output_dir $HOME/state/test/ --checkpoint final.ckpt
    ```
    ```bash
    state tx infer --output $HOME/state/test/ --output_dir /path/to/model/ --checkpoint /path/to/model/final.ckpt --adata /path/to/anndata/processed.h5 --pert_col gene --embed_key X_hvg
    ```

### Data Preprocessing for ST Models

Prepare your data for training and inference.

#### Training Data Preprocessing

Prepare your data for training using the following:

```bash
state tx preprocess_train \
  --adata /path/to/raw_data.h5ad \
  --output /path/to/preprocessed_training_data.h5ad \
  --num_hvgs 2000
```

This performs normalization, log-transformation, and HVG selection.

#### Inference Data Preprocessing

Create a control template for inference:

```bash
state tx preprocess_infer \
  --adata /path/to/real_data.h5ad \
  --output /path/to/control_template.h5ad \
  --control_condition "DMSO" \
  --pert_col "treatment" \
  --seed 42
```

### TOML Configuration for ST Models

Configure your experiments using TOML files to specify datasets, training splits, and evaluation scenarios.

**Key Configuration Sections:**

*   `[datasets]`: Defines dataset paths.
*   `[training]`: Specifies datasets used for training.
*   `[zeroshot]`: Holds entire cell types for validation/testing.
*   `[fewshot]`: Defines perturbation-level splits within cell types.

Refer to the original documentation for specific examples and details.

## State Embedding Model (SE) - Generating Cell Embeddings

SE models generate embeddings for single-cell data.

1.  **Training:** Train an SE model using a configuration file:

    ```bash
    state emb fit --conf ${CONFIG}
    ```

2.  **Inference:** Run inference with a trained checkpoint:

    ```bash
    state emb transform \
      --model-folder /large_storage/ctc/userspace/aadduri/SE-600M \
      --checkpoint /large_storage/ctc/userspace/aadduri/SE-600M/se600m_epoch15.ckpt \
      --input /large_storage/ctc/datasets/replogle/rpe1_raw_singlecell_01.h5ad \
      --output /home/aadduri/vci_pretrain/test_output.h5ad
    ```
    *   Note: Requires CSR matrix format and `gene_name` in the `var` dataframe of the input `.h5ad` file.

### Vector Database (LanceDB) for SE

Leverage LanceDB for efficient similarity search and analysis of embeddings.

1.  **Install Dependencies:**

    ```bash
    uv tool install ".[vectordb]"
    ```
    (If needed: `uv sync --extra vectordb`)

2.  **Build the Vector Database:**

    ```bash
    state emb transform \
      --model-folder /large_storage/ctc/userspace/aadduri/SE-600M \
      --input /large_storage/ctc/public/scBasecamp/GeneFull_Ex50pAS/GeneFull_Ex50pAS/Homo_sapiens/SRX27532045.h5ad \
      --lancedb tmp/state_embeddings.lancedb \
      --gene-column gene_symbols
    ```
    Running multiple times appends to the database.
3.  **Query the Database:**

    *   Obtain embeddings for query data:

        ```bash
        state emb transform \
          --model-folder /large_storage/ctc/userspace/aadduri/SE-600M \
          --input /large_storage/ctc/public/scBasecamp/GeneFull_Ex50pAS/GeneFull_Ex50pAS/Homo_sapiens/SRX27532046.h5ad \
          --output tmp/SRX27532046.h5ad \
          --gene-column gene_symbols
        ```

    *   Query the database:

        ```bash
        state emb query \
          --lancedb tmp/state_embeddings.lancedb \
          --input tmp/SRX27532046.h5ad \
          --output tmp/similar_cells.csv \
          --k 3
        ```

## Singularity Container

Build and run the STATE container:

*   Build the container:

    ```bash
    singularity build state.sif singularity.def
    ```

*   Run the container:

    ```bash
    singularity run state.sif --help
    ```

    Example:

    ```bash
    singularity run --nv -B /large_storage:/large_storage \
      state.sif emb transform \
        --model-folder /large_storage/ctc/userspace/aadduri/SE-600M \
        --checkpoint /large_storage/ctc/userspace/aadduri/SE-600M/se600m_epoch15.ckpt \
        --input /large_storage/ctc/datasets/replogle/rpe1_raw_singlecell_01.h5ad \
        --output test_output.h5ad
    ```

## Licenses

*   Code: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)](LICENSE)
*   Model Weights & Output: [Arc Research Institute State Model Non-Commercial License](MODEL_LICENSE.md), subject to the [Arc Research Institute State Model Acceptable Use Policy](MODEL_ACCEPTABLE_USE_POLICY.md).

**Citation:**  Please cite the State [paper](https://arcinstitute.org/manuscripts/State) in publications using this code or model parameters.