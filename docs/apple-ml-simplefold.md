<div align="center">
  <h1>SimpleFold: Revolutionizing Protein Folding with Simplicity</h1>
  <p><em>Unlocking protein structure prediction with a novel transformer-based approach.</em></p>
  <p>
    <a href="https://github.com/apple/ml-simplefold">
      <img src="https://img.shields.io/badge/GitHub-SimpleFold-blue?style=flat&logo=github" alt="GitHub">
    </a>
    <br>
    <a href="https://arxiv.org/abs/2509.18480">Paper</a> | <a href="#citation">BibTex</a>
  </p>
  <img src="assets/intro.png" width="750" alt="SimpleFold Illustration">
</div>

**SimpleFold is a groundbreaking protein folding model that leverages the power of standard transformer layers, offering a simpler yet highly effective approach to protein structure prediction.** This repository provides the code and resources to reproduce and build upon the research presented in the paper [*SimpleFold: Folding Proteins is Simpler than You Think*](https://arxiv.org/abs/2509.18480).

**Key Features:**

*   **Transformer-Based Architecture:** SimpleFold uses only general-purpose transformer layers, eliminating the need for complex, domain-specific modules.
*   **Generative Flow-Matching:** Trained with a generative flow-matching objective, enabling strong ensemble prediction performance.
*   **Scalable and High-Performing:** SimpleFold models are trained with up to 3 billion parameters, achieving state-of-the-art results on standard folding benchmarks.
*   **Open-Source and Accessible:** Provides clear installation instructions, example usage, and evaluation tools.
*   **Supports Multiple Backends:** Inference is supported using both PyTorch and MLX (optimized for Apple hardware).

## Installation

Get started by cloning the repository and installing the necessary dependencies:

```bash
git clone https://github.com/apple/ml-simplefold.git
cd ml-simplefold
conda create -n simplefold python=3.10
python -m pip install -U pip build; pip install -e .
```

**For MLX backend on Apple silicon:**

```bash
pip install mlx==0.28.0
pip install git+https://github.com/facebookresearch/esm.git
```

## Example Usage

A sample Jupyter notebook (`sample.ipynb`) is provided to demonstrate protein structure prediction from protein sequences.

## Inference

Predict protein structures from FASTA files using the command line tool.  Specify model size, inference parameters, and output directory.

```bash
simplefold \
    --simplefold_model simplefold_100M \  # specify folding model in simplefold_100M/360M/700M/1.1B/1.6B/3B
    --num_steps 500 --tau 0.01 \        # specify inference setting
    --nsample_per_protein 1 \           # number of generated conformers per target
    --plddt \                           # output pLDDT
    --fasta_path [FASTA_PATH] \         # path to the target fasta directory or file
    --output_dir [OUTPUT_DIR] \         # path to the output directory
    --backend [mlx, torch]              # choose from MLX and PyTorch for inference backend 
```

## Evaluation

Pre-computed structure predictions for various datasets are available for evaluation:

*   [CAMEO22](https://ml-site.cdn-apple.com/models/simplefold/cameo22_predictions.zip)
*   [CASP14](https://ml-site.cdn-apple.com/models/simplefold/casp14_predictions.zip)
*   [Apo](https://ml-site.cdn-apple.com/models/simplefold/apo_predictions.zip)
*   [CoDNaS](https://ml-site.cdn-apple.com/models/simplefold/codnas_predictions.zip)

Use the provided scripts and the [openstructure](https://git.scicore.unibas.ch/schwede/openstructure/) Docker image to analyze folding results.

```bash
# For folding tasks (CASP14/CAMEO22)
python src/simplefold/evaluation/analyze_folding.py \
    --data_dir [PATH_TO_TARGET_MMCIF] \
    --sample_dir [PATH_TO_PREDICTED_MMCIF] \
    --out_dir [PATH_TO_OUTPUT] \
    --max-workers [NUMBER_OF_WORKERS]

# For two-state prediction (Apo/CoDNaS)
python src/simplefold/evaluation/analyze_two_state.py \
    --data_dir [PATH_TO_TARGET_DATA_DIRECTORY] \
    --sample_dir [PATH_TO_PREDICTED_PDB] \
    --tm_bin [PATH_TO_TMscore_BINARY] \
    --task apo \ # choose from apo and codnas
    --nsample 5
```

## Training

Detailed instructions are provided to train or fine-tune SimpleFold models, including data preparation and training configurations.

### Data Preparation

*   **Training Datasets:** The model is trained on experimental PDB structures, as well as distilled predictions from AFDB SwissProt and AFESM.  Links to filtered target lists are provided.
*   **Processing MMCIF Files:** Instructions are given for processing mmcif files using Redis.

### Training Process

*   Utilize the provided example training configuration (`configs/experiment/train`) with Hydra.
*   Commands are provided to initiate training and use FSDP strategy.

```bash
python train experiment=train
```
```bash
python train_fsdp.py experiment=train_fsdp
```

## Citation

If you use SimpleFold in your research, please cite the following paper:

```
@article{simplefold,
  title={SimpleFold: Folding Proteins is Simpler than You Think},
  author={Wang, Yuyang and Lu, Jiarui and Jaitly, Navdeep and Susskind, Josh and Bautista, Miguel Angel},
  journal={arXiv preprint arXiv:2509.18480},
  year={2025}
}
```

## Acknowledgements

See [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for a list of open-source contributions used in this project.

## License

Please review the project's [LICENSE](LICENSE) and [LICENSE_MODEL](LICENSE_MODEL) files.