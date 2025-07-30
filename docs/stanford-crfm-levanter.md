# Levanter: Train Large Language Models with Ease and Speed

**Levanter is a flexible and efficient framework for training large language models (LLMs) and other foundation models, offering scalability, reproducibility, and ease of use.** [Check out the original repo](https://github.com/stanford-crfm/levanter)!

[![Build Status](https://img.shields.io/github/actions/workflow/status/stanford-crfm/levanter/run_tests.yaml?branch=main)](https://github.com/stanford-crfm/levanter/actions?query=branch%3Amain++)
[![Documentation Status](https://readthedocs.org/projects/levanter/badge/?version=latest)](https://levanter.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/stanford-crfm/levanter?color=blue)](https://github.com/stanford-crfm/levanter/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/levanter?color=blue)](https://pypi.org/project/levanter/)

Levanter is built with JAX, Equinox, and Haliax, providing a powerful yet accessible platform for LLM development.

## Key Features

*   **Legible Code:** Leverages Haliax for easy-to-follow, composable deep learning code.
*   **Scalability:** Designed to train large models on various hardware, including GPUs and TPUs.
*   **Reproducibility:** Ensures bitwise deterministic results for consistent training outcomes.
*   **Distributed Training:** Supports distributed training on TPUs and GPUs, including FSDP and tensor parallelism.
*   **Hugging Face Compatibility:** Seamless integration with the Hugging Face ecosystem for models, tokenizers, and datasets.
*   **High Performance:** Achieves performance comparable to commercial frameworks.
*   **Resilient Checkpointing:** Offers fast, distributed checkpointing and resume functionality.
*   **Cached Data Preprocessing:** Accelerates training by caching preprocessing results.
*   **Rich Logging:** Provides comprehensive metrics and supports various logging backends like WandB and TensorBoard.
*   **Optimizers:** Supports the Sophia optimizer and Optax for advanced optimization strategies.
*   **Flexible Data Handling:** Enables tuning data mixtures without retokenization or shuffling.

## Getting Started

### Installation

Install Levanter using pip:

```bash
pip install levanter
```

For the latest development version:

```bash
pip install git+https://github.com/stanford-crfm/levanter.git
wandb login  # optional, for logging
```

Follow the [Installation Guide](docs/Installation.md) for more detailed instructions.

### Training Examples

*   **Train a GPT2-nano:**

    ```bash
    python -m levanter.main.train_lm --config_path config/gpt2_nano.yaml
    ```

*   **Train a Llama-small on your own data:**

    ```bash
    python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.id openwebtext
    ```

    Customize your training runs with the [Configuration Guide](doc./reference/Configuration.md) and experiment with other supported architectures.

## Supported Architectures

Levanter currently supports:

*   GPT-2
*   LLama (1, 2, and 3)
*   Gemma (1, 2, and 3)
*   Qwen2
*   Qwen3
*   Mistral
*   Mixtral
*   Olmo2
*   Whisper (for speech)

## Distributed and Cloud Training

*   **TPU Cloud VM:**  Refer to the [TPU Getting Started](docs/Getting-Started-TPU-VM.md) guide.
*   **CUDA:** Refer to the [CUDA Getting Started](docs/Getting-Started-GPU.md) guide.

## Contributing

We welcome contributions!  See [CONTRIBUTING.md](CONTRIBUTING.md) and our [issue tracker](https://github.com/stanford-crfm/levanter/issues) for more information.

## License

Levanter is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.