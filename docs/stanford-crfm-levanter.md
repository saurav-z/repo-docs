# Levanter: Train Large Language Models with Speed, Scalability, and Reproducibility

Levanter is a framework for training large language models (LLMs) built with JAX, offering cutting-edge performance and ease of use, as seen by its adoption by Stanford's CRFM. Check out the original repo at [https://github.com/stanford-crfm/levanter](https://github.com/stanford-crfm/levanter).

[![Build Status](https://img.shields.io/github/actions/workflow/status/stanford-crfm/levanter/run_tests.yaml?branch=main)](https://github.com/stanford-crfm/levanter/actions?query=branch%3Amain++)
[![Documentation Status](https://readthedocs.org/projects/levanter/badge/?version=latest)](https://levanter.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/stanford-crfm/levanter?color=blue)](https://github.com/stanford-crfm/levanter/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/levanter?color=blue)](https://pypi.org/project/levanter/)

<!--levanter-intro-start-->

> *You could not prevent a thunderstorm, but you could use the electricity; you could not direct the wind, but you could trim your sail so as to propel your vessel as you pleased, no matter which way the wind blew.* <br/>
> — Cora L. V. Hatch

Levanter empowers researchers and engineers to train LLMs with legibility, scalability, and reproducibility. Built upon JAX, Equinox, and Haliax, Levanter offers a streamlined approach to complex model training.

**Key Features:**

*   **Legible Code:** Uses [Haliax](https://github.com/stanford-crfm/haliax) for intuitive and composable deep learning code.
*   **Scalable Training:** Supports training on various hardware, including GPUs and TPUs, for large-scale models.
*   **Reproducible Results:** Ensures consistent results across runs with bitwise determinism.
*   **Distributed Training:** Supports distributed training on TPUs and GPUs, including FSDP and tensor parallelism.
*   **Hugging Face Integration:** Seamlessly imports and exports models to/from Hugging Face ecosystem, and supports tokenizers, datasets, and models via [SafeTensors](https://github.com/huggingface/safetensors).
*   **High Performance:** Offers performance comparable to established frameworks.
*   **Robust Checkpointing:** Features fast, distributed checkpointing and resume capabilities, minimizing downtime.
*   **Cached Data Preprocessing:** Speeds up training with cached preprocessing for faster resumption and subsequent runs.
*   **Comprehensive Logging:** Logs detailed metrics and supports logging backends like [WandB](https://wandb.ai/site) and [TensorBoard](https://www.tensorflow.org/tensorboard).
*   **Optimization:** Supports the [Sophia](https://arxiv.org/abs/2305.14342) optimizer and [Optax](https://github.com/deepmind/optax).
*   **Flexible Data Handling:** Enables tuning data mixtures without requiring data retokenization.

<!--levanter-intro-end-->

Levanter is developed by the [Stanford Center for Research on Foundation Models (CRFM)](https://crfm.stanford.edu/). Join the community in the #levanter channel on the unofficial [Jax LLM Discord](https://discord.gg/CKazXcbbBm).

## Getting Started

Get up and running quickly with these examples. Refer to the [Getting Started](./docs/Getting-Started-Training.md) guide or the [In-Depth Configuration Guide](doc./reference/Configuration.md) for detailed configuration options, or use `--help`.

### Installation

After installing [JAX](https://github.com/google/jax/blob/main/README.md#installation), install Levanter:

```bash
pip install levanter
```

Or install the latest GitHub version:

```bash
pip install git+https://github.com/stanford-crfm/levanter.git
wandb login  # optional
```

For Haliax and Levanter development, you can install both locally:
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

See the [Installation Guide](docs/Installation.md) for more details.

For TPU setup, refer to the [TPU Getting Started](docs/Getting-Started-TPU-VM.md). GPU support documentation is [here](docs/Getting-Started-GPU.md).

<!--levanter-user-guide-start-->

### Train a GPT2-nano Model

Train a small GPT-2 model:

```bash
python -m levanter.main.train_lm --config_path config/gpt2_nano.yaml
```

Or, if not installed with `-e`:

```bash
python -m levanter.main.train_lm --config_path gpt2_nano
```

This uses the [WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) dataset.

### Train a Llama-small on Custom Data

Specify a Hugging Face dataset:

```bash
python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.id openwebtext
```

Specify a tokenizer and cache directory:

```bash
python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.id openwebtext --data.tokenizer "NousResearch/Llama-2-7b-hf" --data.cache_dir "gs://path/to/cache/dir"
```

Or, use URL-based data:

```bash
python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.train_urls ["https://path/to/train/data_*.jsonl.gz"] --data.validation_urls ["https://path/to/val/data_*.jsonl.gz"]
```

### Customizing Config Files

Modify configuration files for model, dataset, and training parameters. Here's a sample `llama_small_fast.yaml`:

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

Currently, Levanter supports:

*   GPT-2
*   [LLama](https://ai.meta.com/llama/) (1, 2, and 3)
*   [Gemma](https://ai.google.dev/gemma)
*   [Qwen2](https://huggingface.co/Qwen/Qwen2.5-7B)
*   [Qwen3](https://huggingface.co/Qwen/Qwen3-8B)
*   [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
*   [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
*   [Olmo2](https://huggingface.co/allenai/Olmo-2-1124-7B)

For speech, [Whisper](https://huggingface.co/openai/whisper-large-v3) is supported.

#### Continued Pretraining with Llama

Pretrain a Llama 1 or 2 model:

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

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).