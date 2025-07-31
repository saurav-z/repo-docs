# Levanter: Train Large Language Models with Ease and Reproducibility

**Levanter is your go-to framework for training large language models (LLMs) and other foundation models, offering legibility, scalability, and reproducible results.** ([View the Original Repository](https://github.com/stanford-crfm/levanter))

[![Build Status](https://img.shields.io/github/actions/workflow/status/stanford-crfm/levanter/run_tests.yaml?branch=main)](https://github.com/stanford-crfm/levanter/actions?query=branch%3Amain++)
[![Documentation Status](https://readthedocs.org/projects/levanter/badge/?version=latest)](https://levanter.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/stanford-crfm/levanter?color=blue)](https://github.com/stanford-crfm/levanter/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/levanter?color=blue)](https://pypi.org/project/levanter/)

Levanter empowers you to harness the power of large language models with a focus on clarity, performance, and reliability.

**Key Features:**

*   **Legible Code:** Built with [Haliax](https://github.com/stanford-crfm/haliax) for easy-to-follow and composable deep learning code.
*   **Scalable Training:** Supports training on various hardware, including GPUs and TPUs, and scales to accommodate large models.
*   **Reproducible Results:** Bitwise deterministic on TPU, ensuring consistent results across runs.
*   **Distributed Training:** Supports FSDP and tensor parallelism for distributed training on TPUs and GPUs.
*   **Hugging Face Integration:** Seamlessly imports and exports models from the Hugging Face ecosystem, including tokenizers, datasets, and models via [SafeTensors](https://github.com/huggingface/safetensors).
*   **High Performance:** Provides performance comparable to commercially-backed frameworks.
*   **Resilient Checkpointing:** Offers fast, distributed checkpointing and resume functionality to minimize downtime.
*   **Cached Data Preprocessing:** Speeds up training through cached online preprocessing.
*   **Rich Logging:** Comprehensive logging capabilities with support for various backends like [WandB](https://wandb.ai/site) and [TensorBoard](https://www.tensorflow.org/tensorboard/), with the ability to log within JAX `jit`-ted functions.
*   **Optimizations:** Supports the [Sophia](https://arxiv.org/abs/2305.14342) optimizer and [Optax](https://github.com/deepmind/optax) for efficient training.
*   **Flexible Data Mixtures:** Allows tuning data mixtures without the need to retokenize or shuffle data.

Levanter is built upon [JAX](https://github.com/jax-ml/jax), [Equinox](https://github.com/patrick-kidger/equinox), and [Haliax](https://github.com/stanford-crfm/haliax).

## Documentation

*   **Levanter Documentation:** [levanter.readthedocs.io](https://levanter.readthedocs.io/en/latest/)
*   **Haliax Documentation:** [haliax.readthedocs.io](https://haliax.readthedocs.io/en/latest/)

## Getting Started

### Installation

After [installing JAX](https://github.com/google/jax/blob/main/README.md#installation) with the appropriate configuration
for your platform, you can install Levanter with:

```bash
pip install levanter
```

or using the latest version from GitHub:

```bash
pip install git+https://github.com/stanford-crfm/levanter.git
wandb login  # optional, we use wandb for logging
```

If you're developing Haliax and Levanter at the same time, you can do something like.
```bash
git clone https://github.com/stanford-crfm/levanter.git
cd levanter
pip install -e .
cd ..
git clone https://github.com/stanford-crfm/haliax.git
cd haliax
pip install -e .
cd ../levanter
```

Please refer to the [Installation Guide](docs/Installation.md) for more information on how to install Levanter.

If you're using a TPU, more complete documentation for setting that up is available [here](docs/Getting-Started-TPU-VM.md). GPU support is still in-progress; documentation is available [here](docs/Getting-Started-GPU.md).

### Training Examples

#### Training a GPT2-nano

```bash
python -m levanter.main.train_lm --config_path config/gpt2_nano.yaml

# alternatively, if you didn't use -e and are in a different directory
python -m levanter.main.train_lm --config_path gpt2_nano
```

#### Training a Llama-small on your own data

```bash
python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.id openwebtext

# optionally, you may specify a tokenizer and/or a cache directory, which may be local or on gcs
python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.id openwebtext --data.tokenizer "NousResearch/Llama-2-7b-hf" --data.cache_dir "gs://path/to/cache/dir"
```

### Customizing a Config File

Example `llama_small_fast.yaml` file:

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

Levanter currently supports:

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

*   **TPU Cloud VM:** Refer to the [TPU Getting Started](docs/Getting-Started-TPU-VM.md) guide.
*   **CUDA:** Refer to the [CUDA Getting Started](docs/Getting-Started-GPU.md) guide.

## Contributing

[![GitHub repo Good Issues for newbies](https://img.shields.io/github/issues/stanford-crfm/levanter/good%20first%20issue?style=flat&logo=github&logoColor=green&label=Good%20First%20issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) [![GitHub Help Wanted issues](https://img.shields.io/github/issues/stanford-crfm/levanter/help%20wanted?style=flat&logo=github&logoColor=b545d1&label=%22Help%20Wanted%22%20issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) [![GitHub Help Wanted PRs](https://img.shields.io/github/issues-pr/stanford-crfm/levanter/help%20wanted?style=flat&logo=github&logoColor=b545d1&label=%22Help%20Wanted%22%20PRs)](https://github.com/stanford-crfm/levanter/pulls?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) [![GitHub repo Issues](https://img.shields.io/github/issues/stanford-crfm/levanter?style=flat&logo=github&logoColor=red&label=Issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen)

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Levanter is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.