# Levanter: Train Large Language Models with Speed, Scalability, and Reproducibility

Levanter is a framework designed for training large language models (LLMs) and other foundation models, built by the Stanford Center for Research on Foundation Models (CRFM), offering efficient training, reproducibility, and compatibility with popular tools. Check out the original repo at [https://github.com/stanford-crfm/levanter](https://github.com/stanford-crfm/levanter).

[![Build Status](https://img.shields.io/github/actions/workflow/status/stanford-crfm/levanter/run_tests.yaml?branch=main)](https://github.com/stanford-crfm/levanter/actions?query=branch%3Amain++)
[![Documentation Status](https://readthedocs.org/projects/levanter/badge/?version=latest)](https://levanter.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/stanford-crfm/levanter?color=blue)](https://github.com/stanford-crfm/levanter/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/levanter?color=blue)](https://pypi.org/project/levanter/)

> *You could not prevent a thunderstorm, but you could use the electricity; you could not direct the wind, but you could trim your sail so as to propel your vessel as you pleased, no matter which way the wind blew.* <br/>
> — Cora L. V. Hatch

## Key Features

*   **Legible Code:** Utilizes [Haliax](https://github.com/stanford-crfm/haliax) for easy-to-follow, composable deep learning code.
*   **Scalability:** Designed to train large models efficiently on various hardware, including GPUs and TPUs.
*   **Reproducibility:** Ensures bitwise deterministic results for consistent training outcomes.
*   **Distributed Training:** Supports distributed training on TPUs and GPUs, including FSDP and tensor parallelism.
*   **Hugging Face Compatibility:** Seamless integration with the Hugging Face ecosystem for model importing/exporting, tokenizers, and datasets via [SafeTensors](https://github.com/huggingface/safetensors).
*   **High Performance:** Achieves performance comparable to commercially-backed frameworks.
*   **Resilience:** Offers fast, distributed checkpointing and rapid resumption, minimizing downtime due to preemption or hardware failures.
*   **Cached On-Demand Data Preprocessing:** Accelerates training with cached preprocessing results, optimizing resume times.
*   **Rich Logging:** Provides detailed metrics and supports various logging backends such as [WandB](https://wandb.ai/site) and [TensorBoard](https://www.tensorflow.org/tensorboard).
*   **Optimization:** Supports advanced optimizers like [Sophia](https://arxiv.org/abs/2305.14342) and [Optax](https://github.com/deepmind/optax).
*   **Flexibility:** Allows tuning data mixtures without retokenizing or shuffling data.

## Getting Started

### Installation

Install Levanter after setting up your JAX environment:

```bash
pip install levanter
```

Or, install the latest version from GitHub:

```bash
pip install git+https://github.com/stanford-crfm/levanter.git
wandb login  # optional, we use wandb for logging
```

Refer to the [Installation Guide](docs/Installation.md) for detailed instructions.

### Training Examples

*   **Training a GPT2-nano:**

```bash
python -m levanter.main.train_lm --config_path config/gpt2_nano.yaml
```

*   **Training a Llama-small on your own data:**

```bash
python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.id openwebtext
```

### Customizing a Config File

Modify config files to adjust models, datasets, and training parameters.  Here's the `llama_small_fast.yaml` as an example:

```yaml
# (See original README for contents)
```

### Supported Architectures

*   GPT-2
*   LLama (1, 2, and 3)
*   Gemma (1, 2, and 3)
*   Qwen2
*   Qwen3
*   Mistral
*   Mixtral
*   Olmo2
*   Whisper (for speech)

#### Continued Pretraining with Llama

```bash
python -m levanter.main.train_lm --config_path config/llama2_7b_continued.yaml
```

### Distributed and Cloud Training

*   **TPU Cloud VM:** See the [TPU Getting Started](docs/Getting-Started-TPU-VM.md) guide.
*   **CUDA:** See the [CUDA Getting Started](docs/Getting-Started-GPU.md) guide.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

*   [![GitHub repo Good Issues for newbies](https://img.shields.io/github/issues/stanford-crfm/levanter/good%20first%20issue?style=flat&logo=github&logoColor=green&label=Good%20First%20issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
*   [![GitHub Help Wanted issues](https://img.shields.io/github/issues/stanford-crfm/levanter/help%20wanted?style=flat&logo=github&logoColor=b545d1&label=%22Help%20Wanted%22%20issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)
*   [![GitHub Help Wanted PRs](https://img.shields.io/github/issues-pr/stanford-crfm/levanter/help%20wanted?style=flat&logo=github&logoColor=b545d1&label=%22Help%20Wanted%22%20PRs)](https://github.com/stanford-crfm/levanter/pulls?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)
*   [![GitHub repo Issues](https://img.shields.io/github/issues/stanford-crfm/levanter?style=flat&logo=github&logoColor=red&label=Issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen)

## License

Levanter is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.