# Levanter: Train Large Language Models with Speed, Scalability, and Reproducibility

Levanter is a powerful framework for training large language models (LLMs), offering unparalleled legibility, scalability, and deterministic reproducibility, built on JAX and Haliax. **[Explore Levanter on GitHub](https://github.com/stanford-crfm/levanter)**

[![Build Status](https://img.shields.io/github/actions/workflow/status/stanford-crfm/levanter/run_tests.yaml?branch=main)](https://github.com/stanford-crfm/levanter/actions?query=branch%3Amain++)
[![Documentation Status](https://readthedocs.org/projects/levanter/badge/?version=latest)](https://levanter.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/stanford-crfm/levanter?color=blue)](https://github.com/stanford-crfm/levanter/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/levanter?color=blue)](https://pypi.org/project/levanter/)

> *You could not prevent a thunderstorm, but you could use the electricity; you could not direct the wind, but you could trim your sail so as to propel your vessel as you pleased, no matter which way the wind blew.* <br/>
> — Cora L. V. Hatch

## Key Features

*   **Legible Code:** Built on Haliax for easy-to-follow and composable deep learning code.
*   **Scalability:** Train large models efficiently on GPUs and TPUs.
*   **Reproducibility:** Ensures consistent results with the same configuration, even with preemption and resumption.
*   **Distributed Training:** Supports FSDP and tensor parallelism across TPUs and GPUs.
*   **Hugging Face Compatibility:** Seamlessly integrates with Hugging Face tokenizers, datasets, and models via SafeTensors.
*   **High Performance:** Achieves performance comparable to commercial frameworks.
*   **Resilience:** Offers fast, distributed checkpointing and resume features to avoid data seek.
*   **Cached Data Preprocessing:** Caches online preprocessing for faster resumes and subsequent runs.
*   **Rich Logging:** Provides comprehensive metrics logging with support for WandB, TensorBoard, and custom backends.
*   **Optimization:** Supports the Sophia optimizer and Optax for efficient training.
*   **Flexible Data Handling:** Supports tuning data mixtures without needing to retokenize or shuffle data.

## Getting Started

### Installation

After installing JAX, install Levanter using pip:

```bash
pip install levanter
```

For the latest version:

```bash
pip install git+https://github.com/stanford-crfm/levanter.git
wandb login  # optional
```

For detailed installation instructions, see the [Installation Guide](docs/Installation.md).  For TPU setup, refer to the [TPU Getting Started Guide](docs/Getting-Started-TPU-VM.md) and the [GPU Getting Started Guide](docs/Getting-Started-GPU.md).

### Training Examples

**Training a GPT2-nano:**

```bash
python -m levanter.main.train_lm --config_path config/gpt2_nano.yaml
```

**Training a Llama-small on your own data:**

```bash
python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.id openwebtext
```

Or, specify your dataset URLs:

```bash
python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.train_urls ["https://path/to/train/data_*.jsonl.gz"] --data.validation_urls ["https://path/to/val/data_*.jsonl.gz"]
```

See [Getting Started](./docs/Getting-Started-Training.md) or the [In-Depth Configuration Guide](doc./reference/Configuration.md) for further configuration options.

### Supported Architectures

Levanter supports the following architectures:

*   GPT-2
*   [LLama](https://ai.meta.com/llama/), including Llama 1, 2 and 3
*   [Gemma](https://ai.google.dev/gemma), including Gemma 1, 2 and Gemma 3.
*   [Qwen2](https://huggingface.co/Qwen/Qwen2.5-7B)
*   [Qwen3](https://huggingface.co/Qwen/Qwen3-8B)
*   [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
*   [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
*   [Olmo2](https://huggingface.co/allenai/Olmo-2-1124-7B)

For speech, we currently only support [Whisper](https://huggingface.co/openai/whisper-large-v3).

#### Continued Pretraining with Llama

Here's an example of how to continue pretraining a Llama 1 or Llama 2 model on the OpenWebText dataset:

```bash
python -m levanter.main.train_lm --config_path config/llama2_7b_continued.yaml
```


## Distributed and Cloud Training

### Training on a TPU Cloud VM

Please see the [TPU Getting Started](docs/Getting-Started-TPU-VM.md) guide for more information on how to set up a TPU Cloud VM and run Levanter there.

### Training with CUDA

Please see the [CUDA Getting Started](docs/Getting-Started-GPU.md) guide for more information on how to set up a CUDA environment and run Levanter there.

## Documentation

*   [Levanter Documentation](https://levanter.readthedocs.io/en/latest/)
*   [Haliax Documentation](https://haliax.readthedocs.io/en/latest/)

## Contributing

We welcome contributions!  See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

[![GitHub repo Good Issues for newbies](https://img.shields.io/github/issues/stanford-crfm/levanter/good%20first%20issue?style=flat&logo=github&logoColor=green&label=Good%20First%20issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
[![GitHub Help Wanted issues](https://img.shields.io/github/issues/stanford-crfm/levanter/help%20wanted?style=flat&logo=github&logoColor=b545d1&label=%22Help%20Wanted%22%20issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)
[![GitHub Help Wanted PRs](https://img.shields.io/github/issues-pr/stanford-crfm/levanter/help%20wanted?style=flat&logo=github&logoColor=b545d1&label=%22Help%20Wanted%22%20PRs)](https://github.com/stanford-crfm/levanter/pulls?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)
[![GitHub repo Issues](https://img.shields.io/github/issues/stanford-crfm/levanter?style=flat&logo=github&logoColor=red&label=Issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen)

## License

Levanter is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.