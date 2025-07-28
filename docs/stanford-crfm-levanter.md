# Levanter: Train Large Language Models with Legibility, Scalability, and Reproducibility

[<img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/stanford-crfm/levanter/run_tests.yaml?branch=main">](https://github.com/stanford-crfm/levanter/actions?query=branch%3Amain++)
[<img alt="Documentation Status" src="https://readthedocs.org/projects/levanter/badge/?version=latest">](https://levanter.readthedocs.io/en/latest/?badge=latest)
[<img alt="License" src="https://img.shields.io/github/license/stanford-crfm/levanter?color=blue" />](LICENSE)
[<img alt="PyPI" src="https://img.shields.io/pypi/v/levanter?color=blue" />](https://pypi.org/project/levanter/)

**Levanter is a powerful framework built by Stanford's CRFM for efficiently training large language models, offering a blend of performance, readability, and reliability.**

Inspired by Cora L. V. Hatch's quote, Levanter empowers you to harness the "electricity" of cutting-edge LLM training while controlling the "sail" of your model's performance:

*   **Legible:** Built with [Haliax](https://github.com/stanford-crfm/haliax), a named tensor library, enabling clear and composable deep learning code.
*   **Scalable:** Designed to train large models across various hardware configurations, including GPUs and TPUs.
*   **Reproducible:** Ensures consistent results across training runs, even with interruptions, thanks to bitwise deterministic behavior.

Built on [JAX](https://github.com/jax-ml/jax), [Equinox](https://github.com/patrick-kidger/equinox), and [Haliax](https://github.com/stanford-crfm/haliax).

## Key Features

*   **Distributed Training:** Supports distributed training on TPUs and GPUs, including FSDP and tensor parallelism.
*   **Hugging Face Compatibility:** Seamlessly integrates with the Hugging Face ecosystem, including tokenizers, datasets, and SafeTensors for model import/export.
*   **High Performance:** Matches or exceeds the performance of commercially-backed frameworks like MosaicML's Composer and Google's MaxText.
*   **Resilient Checkpointing:** Offers fast, distributed checkpointing and resume functionality, protecting against preemption and hardware failures.
*   **Cached Data Preprocessing:** Caches preprocessed data for faster resume times and accelerated subsequent training runs.
*   **Comprehensive Logging:** Provides rich and detailed metrics logging and supports logging backends such as [WandB](https://wandb.ai/site) and [TensorBoard](https://www.tensorflow.org/tensorboard/).
*   **Reproducibility:** (On TPUs) Bitwise deterministic training ensures consistent results across runs.
*   **Distributed Checkpointing:** Supports distributed checkpointing via Google's [TensorStore](https://google.github.io/tensorstore/).
*   **Optimization:** Supports the new [Sophia](https://arxiv.org/abs/2305.14342) optimizer and [Optax](https://github.com/deepmind/optax) with AdamW, etc.
*   **Data Flexibility:** Enables tuning data mixtures without retokenizing or shuffling data.

## Documentation

*   **Levanter Documentation:** [levanter.readthedocs.io](https://levanter.readthedocs.io/en/latest/)
*   **Haliax Documentation:** [haliax.readthedocs.io](https://haliax.readthedocs.io/en/latest/)

## Getting Started

Find out more about training configurations and parameters in the [Getting Started](./docs/Getting-Started-Training.md) guide and the [In-Depth Configuration Guide](doc./reference/Configuration.md).

### Installation

1.  Install [JAX](https://github.com/google/jax/blob/main/README.md#installation).
2.  Install Levanter:

```bash
pip install levanter
wandb login  # optional, for logging
```

For detailed installation instructions, see the [Installation Guide](docs/Installation.md).

### Example Training Commands

**Train a GPT2-nano:**

```bash
python -m levanter.main.train_lm --config_path config/gpt2_nano.yaml
```

**Train a Llama-small on a dataset:**

```bash
python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.id openwebtext
```

**Customize the config file**
You can modify config files like `llama_small_fast.yaml` to change model parameters, dataset, or training settings. See the existing config files for examples.

## Supported Architectures

Levanter currently supports the following architectures:

*   GPT-2
*   [LLama](https://ai.meta.com/llama/) (Llama 1, 2, and 3)
*   [Gemma](https://ai.google.dev/gemma) (Gemma 1, 2, and 3)
*   [Qwen2](https://huggingface.co/Qwen/Qwen2.5-7B)
*   [Qwen3](https://huggingface.co/Qwen/Qwen3-8B)
*   [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
*   [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
*   [Olmo2](https://huggingface.co/allenai/Olmo-2-1124-7B)

For speech, Whisper is also supported.

### Continued Pretraining with Llama

```bash
python -m levanter.main.train_lm --config_path config/llama2_7b_continued.yaml
```

## Distributed and Cloud Training

### Training on a TPU Cloud VM

See the [TPU Getting Started](docs/Getting-Started-TPU-VM.md) guide.

### Training with CUDA

See the [CUDA Getting Started](docs/Getting-Started-GPU.md) guide.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details and find "good first issue" and "help wanted" issues here:

[![GitHub repo Good Issues for newbies](https://img.shields.io/github/issues/stanford-crfm/levanter/good%20first%20issue?style=flat&logo=github&logoColor=green&label=Good%20First%20issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
[![GitHub Help Wanted issues](https://img.shields.io/github/issues/stanford-crfm/levanter/help%20wanted?style=flat&logo=github&logoColor=b545d1&label=%22Help%20Wanted%22%20issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)
[![GitHub Help Wanted PRs](https://img.shields.io/github/issues-pr/stanford-crfm/levanter/help%20wanted?style=flat&logo=github&logoColor=b545d1&label=%22Help%20Wanted%22%20PRs)](https://github.com/stanford-crfm/levanter/pulls?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)
[![GitHub repo Issues](https://img.shields.io/github/issues/stanford-crfm/levanter?style=flat&logo=github&logoColor=red&label=Issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen)

## License

Levanter is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

---
[Back to Top](https://github.com/stanford-crfm/levanter)
```
Key improvements and SEO considerations:

*   **Clear Title and Introduction:** Uses the target keyword "large language models" in the title and immediately introduces the core benefit.
*   **Concise Summary:**  Highlights key value propositions in the first paragraph.
*   **SEO-Friendly Headings:**  Uses clear, descriptive headings and subheadings for better organization and SEO.
*   **Keyword Optimization:** The term "large language models" appears naturally and frequently. Other related terms like "LLM training," "distributed training," and "reproducibility" are included.
*   **Bulleted Feature List:**  Presents key features in an easy-to-read bulleted format.
*   **Call to Action (Implied):**  The "Getting Started" section encourages users to try the framework.
*   **Internal Linking:** Links to key documentation within the README.
*   **External Links:** Includes relevant links to the project's dependencies and related resources.
*   **Clear Structure:** Improved overall readability and flow.
*   **Contribution Section:**  Highlights how to contribute and links to relevant issues, improving community engagement.
*   **"Back to Top" Link:** Added for navigation.
*   **Concise Installation:** Installation instructions are made concise and easy to follow.
*   **Code Blocks with Labels:** The code block are clearly separated and easy to identify.
*   **Comprehensive Architecture List:** The architectures were included for better visibility.