# Levanter: Train Large Language Models with Speed and Reproducibility

> **Levanter** is a cutting-edge framework for training large language models (LLMs) and other foundation models, offering legibility, scalability, and deterministic results for unparalleled performance. [Check out the original repo](https://github.com/stanford-crfm/levanter).

[![Build Status](https://img.shields.io/github/actions/workflow/status/stanford-crfm/levanter/run_tests.yaml?branch=main)](https://github.com/stanford-crfm/levanter/actions?query=branch%3Amain++)
[![Documentation Status](https://readthedocs.org/projects/levanter/badge/?version=latest)](https://levanter.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/stanford-crfm/levanter?color=blue)](https://github.com/stanford-crfm/levanter/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/levanter?color=blue)](https://pypi.org/project/levanter/)

Levanter empowers researchers and developers to train LLMs efficiently and reliably. Built with JAX, Equinox, and Haliax, Levanter provides a robust and flexible platform for your foundation model training needs.

**Key Features:**

*   **Legible Code:** Built upon [Haliax](https://github.com/stanford-crfm/haliax), Levanter uses named tensors for easy-to-follow and composable deep learning code.
*   **Scalability:** Train large models across various hardware, including GPUs and TPUs, with distributed training support.
*   **Reproducibility:** Achieve bitwise deterministic results on TPUs, ensuring consistent outcomes across training runs.
*   **Distributed Training:** Support for distributed training on TPUs and GPUs, including FSDP and tensor parallelism.
*   **Hugging Face Integration:** Seamlessly import and export models, tokenizers, and datasets from the Hugging Face ecosystem using [SafeTensors](https://github.com/huggingface/safetensors).
*   **High Performance:** Enjoy performance that rivals commercially-backed frameworks.
*   **Resilience:** Benefit from fast, distributed checkpointing and resume functionality to ensure robust training, even in the face of hardware failures.
*   **Cached Data Preprocessing:** Accelerate training with cached preprocessing, enabling faster resumes and subsequent runs.
*   **Comprehensive Logging:** Monitor your training with detailed metrics and support for logging backends like [WandB](https://wandb.ai/site) and [TensorBoard](https://www.tensorflow.org/tensorboard).
*   **Optimization Support:** Explore cutting-edge optimizers like [Sophia](https://arxiv.org/abs/2305.14342) and Optax for AdamW and more.
*   **Data Flexibility:** Tune data mixtures without retokenizing or shuffling your data.

## Documentation

*   **Levanter Documentation:** [levanter.readthedocs.io](https://levanter.readthedocs.io/en/latest/)
*   **Haliax Documentation:** [haliax.readthedocs.io](https://haliax.readthedocs.io/en/latest/)

## Getting Started

### Installation

After installing [JAX](https://github.com/google/jax/blob/main/README.md#installation) with the appropriate configuration
for your platform, you can install Levanter with:

```bash
pip install levanter
```

or using the latest version from GitHub:

```bash
pip install git+https://github.com/stanford-crfm/levanter.git
wandb login  # optional, we use wandb for logging
```

For detailed installation steps, see the [Installation Guide](docs/Installation.md).

### Training Examples

Here are some examples to get you started:

#### Training a GPT2-nano

```bash
python -m levanter.main.train_lm --config_path config/gpt2_nano.yaml
```

#### Training a Llama-small on your own data

```bash
python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.id openwebtext
```

#### Customizing a Config File

Here's
the `llama_small_fast.yaml` file:

```yaml
data:
  train_urls:
      - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_train.{1..128}-of-128.jsonl.gz"
  validation_urls:
      - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_val.{1..8}-of-8.jsonl.gz"
  cache_dir: "gs://pubmed-mosaic/tokenized/openwebtext/"
model:
  type: llama
  hidden_dim: 768
  intermediate_dim: 2048
  num_heads: 12
  num_kv_heads: 12
  num_layers: 12
  seq_len: 1024
  gradient_checkpointing: true
trainer:
  tracker:
    type: wandb
    project: "levanter"
    tags: [ "openwebtext", "llama" ]

  mp: p=f32,c=bfloat16
  model_axis_size: 1
  per_device_parallelism: 4

  train_batch_size: 512
optimizer:
  learning_rate: 6E-4
  weight_decay: 0.1
  min_lr_ratio: 0.1
```

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

#### Continued Pretraining with Llama

```bash
python -m levanter.main.train_lm --config_path config/llama2_7b_continued.yaml
```

## Distributed and Cloud Training

### Training on a TPU Cloud VM

See the [TPU Getting Started](docs/Getting-Started-TPU-VM.md) guide.

### Training with CUDA

See the [CUDA Getting Started](docs/Getting-Started-GPU.md) guide.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

[![GitHub repo Good Issues for newbies](https://img.shields.io/github/issues/stanford-crfm/levanter/good%20first%20issue?style=flat&logo=github&logoColor=green&label=Good%20First%20issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) [![GitHub Help Wanted issues](https://img.shields.io/github/issues/stanford-crfm/levanter/help%20wanted?style=flat&logo=github&logoColor=b545d1&label=%22Help%20Wanted%22%20issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) [![GitHub Help Wanted PRs](https://img.shields.io/github/issues-pr/stanford-crfm/levanter/help%20wanted?style=flat&logo=github&logoColor=b545d1&label=%22Help%20Wanted%22%20PRs)](https://github.com/stanford-crfm/levanter/pulls?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) [![GitHub repo Issues](https://img.shields.io/github/issues/stanford-crfm/levanter?style=flat&logo=github&logoColor=red&label=Issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen)

## License

Levanter is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.