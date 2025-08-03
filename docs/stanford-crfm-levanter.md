# Levanter: Train Large Language Models with Ease, Scalability, and Reproducibility

> **Levanter empowers you to train cutting-edge LLMs, offering unparalleled scalability, legibility, and reproducibility.**  Learn more and contribute at the [Levanter GitHub repository](https://github.com/stanford-crfm/levanter).

[![Build Status](https://img.shields.io/github/actions/workflow/status/stanford-crfm/levanter/run_tests.yaml?branch=main)](https://github.com/stanford-crfm/levanter/actions?query=branch%3Amain++)
[![Documentation Status](https://readthedocs.org/projects/levanter/badge/?version=latest)](https://levanter.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/stanford-crfm/levanter?color=blue)](https://github.com/stanford-crfm/levanter/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/levanter?color=blue)](https://pypi.org/project/levanter/)

<!--levanter-intro-start-->

Levanter is a powerful framework built for training large language models (LLMs) and other foundation models. It is designed for legibility, scalability, and reproducibility.

## Key Features

*   **Legible Code:** Leverages the [Haliax](https://github.com/stanford-crfm/haliax) named tensor library for easy-to-follow, composable, and high-performance deep learning code.
*   **Scalable Training:** Supports training on various hardware, including GPUs and TPUs, with the ability to scale to large models.
*   **Reproducible Results:** Ensures bitwise deterministic behavior on TPUs, guaranteeing consistent results across runs, even with preemption and resumption.
*   **Distributed Training:** Supports distributed training on TPUs and GPUs, including FSDP and tensor parallelism.
*   **Hugging Face Compatibility:** Seamlessly integrates with the Hugging Face ecosystem for importing and exporting models, tokenizers, datasets, and models via [SafeTensors](https://github.com/huggingface/safetensors).
*   **High Performance:** Delivers performance comparable to commercially-backed frameworks like MosaicML's Composer or Google's MaxText.
*   **Resilient Checkpointing:** Provides fast, distributed checkpointing and fast resume from checkpoints, enhancing robustness against hardware failures and preemption.
*   **Cached Data Preprocessing:** Speeds up training with cached preprocessing results for faster resumes and subsequent runs.
*   **Rich Logging:** Includes detailed metrics logging with support for various backends like [WandB](https://wandb.ai/site) and [TensorBoard](https://www.tensorflow.org/tensorboard) and logging within JAX jit functions.
*   **Optimization:** Supports advanced optimizers like [Sophia](https://arxiv.org/abs/2305.14342) and [Optax](https://github.com/deepmind/optax) for enhanced performance.
*   **Flexible Data Mixtures:** Allows tuning data mixtures without retokenization or data shuffling.

<!--levanter-intro-end-->

Levanter was created by the [Stanford's Center for Research on Foundation Models (CRFM)](https://crfm.stanford.edu/) research engineering team.

## Getting Started

### Installation

Install Levanter after installing JAX (see the [JAX installation guide](https://github.com/google/jax/blob/main/README.md#installation)) with:

```bash
pip install levanter
```

or to install the latest development version:

```bash
pip install git+https://github.com/stanford-crfm/levanter.git
wandb login  # optional, we use wandb for logging
```

For detailed installation guidance, consult the [Installation Guide](docs/Installation.md).

### Training Examples

*   **Training a GPT2-nano:**

    ```bash
    python -m levanter.main.train_lm --config_path config/gpt2_nano.yaml
    ```

*   **Training a Llama-small on your own data:**

    ```bash
    python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.id openwebtext
    ```
    Customize with your data and tokenizer:

    ```bash
    python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.id openwebtext --data.tokenizer "NousResearch/Llama-2-7b-hf" --data.cache_dir "gs://path/to/cache/dir"
    ```

*   **Using URLs for data:**

    ```bash
    python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.train_urls ["https://path/to/train/data_*.jsonl.gz"] --data.validation_urls ["https://path/to/val/data_*.jsonl.gz"]
    ```

### Available Architectures

Levanter currently supports:

*   GPT-2
*   LLama (Llama 1, 2 and 3)
*   Gemma (1, 2 and 3)
*   Qwen2
*   Qwen3
*   Mistral
*   Mixtral
*   Olmo2
*   Whisper (for speech)

### Continued Pretraining

```bash
python -m levanter.main.train_lm --config_path config/llama2_7b_continued.yaml
```

## Training on TPU and GPU

See these guides for setup:

*   [TPU Getting Started](docs/Getting-Started-TPU-VM.md)
*   [CUDA Getting Started](docs/Getting-Started-GPU.md)

## Contributing

We welcome contributions!  See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

[![GitHub repo Good Issues for newbies](https://img.shields.io/github/issues/stanford-crfm/levanter/good%20first%20issue?style=flat&logo=github&logoColor=green&label=Good%20First%20issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
[![GitHub Help Wanted issues](https://img.shields.io/github/issues/stanford-crfm/levanter/help%20wanted?style=flat&logo=github&logoColor=b545d1&label=%22Help%20Wanted%22%20issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)
[![GitHub Help Wanted PRs](https://img.shields.io/github/issues-pr/stanford-crfm/levanter/help%20wanted?style=flat&logo=github&logoColor=b545d1&label=%22Help%20Wanted%22%20PRs)](https://github.com/stanford-crfm/levanter/pulls?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)
[![GitHub repo Issues](https://img.shields.io/github/issues/stanford-crfm/levanter?style=flat&logo=github&logoColor=red&label=Issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen)

## License

Levanter is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.
```
Key improvements and SEO optimizations:

*   **Concise Hook:** The opening sentence is a clear, direct, and compelling summary of the project's purpose.
*   **Clear Headings:** Uses descriptive and well-formatted headings to organize information.
*   **Bulleted Key Features:** Highlights the essential aspects of Levanter in a concise, easy-to-scan format.
*   **SEO Keywords:**  Includes relevant keywords like "LLM", "Large Language Models", "Training", "Scalable", "Reproducible", and key library names.
*   **Links Back:**  Provides a clear link to the original repository and documentation.
*   **Call to Action:** Encourages readers to learn more and contribute.
*   **Reorganized for Clarity:** Sections are logically arranged for better readability.
*   **Concise Language:** The writing is streamlined to convey information effectively.
*   **Included Important Links:**  Kept all the useful links from the original README.