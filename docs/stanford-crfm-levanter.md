<!-- SEO-optimized README -->

# Levanter: Train Large Language Models with Ease

**Levanter is a powerful and flexible framework, developed by Stanford's CRFM, for training large language models (LLMs) and other foundation models, designed for legibility, scalability, and reproducibility.** ([See the original repo](https://github.com/stanford-crfm/levanter))

[![Build Status](https://img.shields.io/github/actions/workflow/status/stanford-crfm/levanter/run_tests.yaml?branch=main)](https://github.com/stanford-crfm/levanter/actions?query=branch%3Amain++)
[![Documentation Status](https://readthedocs.org/projects/levanter/badge/?version=latest)](https://levanter.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/stanford-crfm/levanter?color=blue)](https://github.com/stanford-crfm/levanter/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/levanter?color=blue)](https://pypi.org/project/levanter/)

<!--levanter-intro-start-->
> *You could not prevent a thunderstorm, but you could use the electricity; you could not direct the wind, but you could trim your sail so as to propel your vessel as you pleased, no matter which way the wind blew.* <br/>
> — Cora L. V. Hatch

Levanter empowers researchers and engineers to train LLMs with confidence, offering a streamlined and efficient workflow.

**Key Features:**

*   **Legible Code:** Built with [Haliax](https://github.com/stanford-crfm/haliax) for easy-to-follow, composable, and high-performance deep learning code.
*   **Scalable Training:** Supports training on diverse hardware, including GPUs and TPUs, and large model sizes.
*   **Reproducible Results:** Ensures bitwise deterministic training on TPUs, guaranteeing consistent results across runs.
*   **Distributed Training**: Supports distributed training on TPUs and GPUs, including FSDP and tensor parallelism.
*   **Hugging Face Compatibility**: Seamless integration with the Hugging Face ecosystem for models, tokenizers, and datasets.
*   **Performance**: Performance that rivals frameworks like MosaicML's Composer and Google's MaxText.
*   **Resilience**: Fast, distributed checkpointing and resume from checkpoints with no data seek, making it robust to preemption and hardware failure.
*   **Cached On-Demand Data Preprocessing**: Preprocesses corpora online and caches results for fast resumes and subsequent runs.
*   **Rich Logging**: Detailed logging of metrics and support for various logging backends, including WandB and TensorBoard.
*   **Optimization**: Support for the new [Sophia](https://arxiv.org/abs/2305.14342) optimizer and [Optax](https://github.com/deepmind/optax).
*   **Flexible Data Handling**: Allows tuning data mixtures without requiring data retokenization or shuffling.

Levanter is developed by the [Stanford's Center for Research on Foundation Models (CRFM)](https://crfm.stanford.edu/)'s research engineering team.

## Getting Started

Quickly get started with Levanter using these examples. Refer to the [Getting Started](./docs/Getting-Started-Training.md) guide or the [In-Depth Configuration Guide](doc./reference/Configuration.md) for detailed configuration options. Use `--help` to explore available options.

### Installation

Install Levanter after [installing JAX](https://github.com/google/jax/blob/main/README.md#installation).

```bash
pip install levanter
```

Or install the latest version from GitHub:

```bash
pip install git+https://github.com/stanford-crfm/levanter.git
wandb login  # optional, we use wandb for logging
```

See the [Installation Guide](docs/Installation.md) for more details.
TPU setup documentation is [here](docs/Getting-Started-TPU-VM.md), and GPU support documentation is [here](docs/Getting-Started-GPU.md).

### Training Examples

#### Train a GPT2-nano

```bash
python -m levanter.main.train_lm --config_path config/gpt2_nano.yaml
```

This will train a GPT2-nano model on the [WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) dataset.

#### Train a Llama-small on your own data

```bash
python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.id openwebtext
```

To use a Hugging Face dataset, set the `data.id`. For custom data URLs:

```bash
python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.train_urls ["https://path/to/train/data_*.jsonl.gz"] --data.validation_urls ["https://path/to/val/data_*.jsonl.gz"]
```

### Config File Customization

Modify the config file to change model, dataset, and training parameters.

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
*   LLama (1, 2, and 3)
*   Gemma
*   Qwen2, Qwen3
*   Mistral, Mixtral
*   Olmo2

And Whisper for speech.

#### Continued Pretraining with Llama

```bash
python -m levanter.main.train_lm --config_path config/llama2_7b_continued.yaml
```

## Distributed and Cloud Training

### TPU Cloud VM Training

See the [TPU Getting Started](docs/Getting-Started-TPU-VM.md) guide.

### CUDA Training

See the [CUDA Getting Started](docs/Getting-Started-GPU.md) guide.

<!--levanter-user-guide-end-->

## Contributing

[![GitHub repo Good Issues for newbies](https://img.shields.io/github/issues/stanford-crfm/levanter/good%20first%20issue?style=flat&logo=github&logoColor=green&label=Good%20First%20issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) [![GitHub Help Wanted issues](https://img.shields.io/github/issues/stanford-crfm/levanter/help%20wanted?style=flat&logo=github&logoColor=b545d1&label=%22Help%20Wanted%22%20issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) [![GitHub Help Wanted PRs](https://img.shields.io/github/issues-pr/stanford-crfm/levanter/help%20wanted?style=flat&logo=github&logoColor=b545d1&label=%22Help%20Wanted%22%20PRs)](https://github.com/stanford-crfm/levanter/pulls?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) [![GitHub repo Issues](https://img.shields.io/github/issues/stanford-crfm/levanter?style=flat&logo=github&logoColor=red&label=Issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen)

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Levanter is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).