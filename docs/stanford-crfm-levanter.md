# Levanter: Train Large Language Models with Ease and Efficiency

[![Build Status](https://img.shields.io/github/actions/workflow/status/stanford-crfm/levanter/run_tests.yaml?branch=main)](https://github.com/stanford-crfm/levanter/actions?query=branch%3Amain++)
[![Documentation Status](https://readthedocs.org/projects/levanter/badge/?version=latest)](https://levanter.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/stanford-crfm/levanter?color=blue)](https://github.com/stanford-crfm/levanter/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/levanter?color=blue)](https://pypi.org/project/levanter/)

Levanter is a powerful and versatile framework, developed by [Stanford's Center for Research on Foundation Models (CRFM)](https://crfm.stanford.edu/), designed for training large language models (LLMs) and other foundation models, offering unparalleled scalability, reproducibility, and ease of use.  **Check out the source code on [GitHub](https://github.com/stanford-crfm/levanter)!**

> *You could not prevent a thunderstorm, but you could use the electricity; you could not direct the wind, but you could trim your sail so as to propel your vessel as you pleased, no matter which way the wind blew.*
> — Cora L. V. Hatch

Levanter provides a streamlined approach to LLM training, focusing on:

*   **Legibility:** Uses the [Haliax](https://github.com/stanford-crfm/haliax) named tensor library for clear, composable, and high-performance deep learning code.
*   **Scalability:** Designed to train large models on a variety of hardware, including GPUs and TPUs, with distributed training support.
*   **Reproducibility:** Ensures consistent results across training runs with bitwise determinism on TPUs, even with preemption and resumption.

Built with JAX, Equinox, and Haliax, Levanter empowers researchers and engineers to train and experiment with cutting-edge models.

## Key Features

*   **Distributed Training:** Supports distributed training on TPUs and GPUs, including FSDP and tensor parallelism.
*   **Hugging Face Integration:** Seamlessly integrates with the Hugging Face ecosystem for model import/export, tokenizers, and datasets (via SafeTensors).
*   **Performance:** Delivers performance that rivals commercially-backed frameworks like MosaicML's Composer or Google's MaxText.
*   **Resilience:** Provides fast, distributed checkpointing and resumption for robust training against failures.
*   **Cached Data Preprocessing:** Caches preprocessed data for faster resumption and subsequent runs.
*   **Comprehensive Logging:** Offers a rich set of metrics logging, supporting backends like WandB and TensorBoard, and the ability to log inside JAX `jit`-ted functions.
*   **Optimization:**  Supports the new Sophia optimizer (up to 2x faster than Adam) and Optax for AdamW and other optimization methods.
*   **Flexible Data Handling:** Allows tuning of data mixtures without retokenization or shuffling.

## Getting Started

### Installation

Install Levanter after [installing JAX](https://github.com/google/jax/blob/main/README.md#installation) with the appropriate configuration
for your platform:

```bash
pip install levanter
```

For the latest version:

```bash
pip install git+https://github.com/stanford-crfm/levanter.git
wandb login  # optional, we use wandb for logging
```

For development, refer to the instructions in the original README.

For more information on installation, see the [Installation Guide](docs/Installation.md).  For TPU setup, see the [TPU Getting Started](docs/Getting-Started-TPU-VM.md) guide. GPU support is documented [here](docs/Getting-Started-GPU.md).

### Training Examples

Here are some examples to get you started:

*   **Train a GPT2-nano:**

    ```bash
    python -m levanter.main.train_lm --config_path config/gpt2_nano.yaml
    ```

*   **Train a Llama-small with your own data:**

    ```bash
    python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.id openwebtext

    # Optionally, specify a tokenizer and cache directory
    python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.id openwebtext --data.tokenizer "NousResearch/Llama-2-7b-hf" --data.cache_dir "gs://path/to/cache/dir"
    ```

    Using data URLs:

    ```bash
    python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.train_urls ["https://path/to/train/data_*.jsonl.gz"] --data.validation_urls ["https://path/to/val/data_*.jsonl.gz"]
    ```

### Configuration

Modify the config files to change the model, dataset, and training parameters.  See the `llama_small_fast.yaml` file for an example and details.

### Supported Architectures

*   GPT-2
*   [LLama](https://ai.meta.com/llama/), including Llama 1, 2 and 3
*   [Gemma](https://ai.google.dev/gemma), including Gemma 1, 2 and Gemma 3.
*   [Qwen2](https://huggingface.co/Qwen/Qwen2.5-7B)
*   [Qwen3](https://huggingface.co/Qwen/Qwen3-8B)
*   [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
*   [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
*   [Olmo2](https://huggingface.co/allenai/Olmo-2-1124-7B)

For speech, we currently only support [Whisper](https://huggingface.co/openai/whisper-large-v3).

### Continuing Pretraining

```bash
python -m levanter.main.train_lm --config_path config/llama2_7b_continued.yaml
```

## Distributed and Cloud Training

Instructions for training on a TPU Cloud VM and with CUDA can be found in the provided guides.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

*   [![GitHub repo Good Issues for newbies](https://img.shields.io/github/issues/stanford-crfm/levanter/good%20first%20issue?style=flat&logo=github&logoColor=green&label=Good%20First%20issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
*   [![GitHub Help Wanted issues](https://img.shields.io/github/issues/stanford-crfm/levanter/help%20wanted?style=flat&logo=github&logoColor=b545d1&label=%22Help%20Wanted%22%20issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)
*   [![GitHub Help Wanted PRs](https://img.shields.io/github/issues-pr/stanford-crfm/levanter/help%20wanted?style=flat&logo=github&logoColor=b545d1&label=%22Help%20Wanted%22%20PRs)](https://github.com/stanford-crfm/levanter/pulls?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)
*   [![GitHub repo Issues](https://img.shields.io/github/issues/stanford-crfm/levanter?style=flat&logo=github&logoColor=red&label=Issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen)

## License

Levanter is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.