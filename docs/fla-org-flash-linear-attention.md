<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-17-orange.svg?style=flat-square)](https://github.com/fla-org/flash-linear-attention/graphs/contributors)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

# Flash Linear Attention: High-Performance Implementations of Linear Attention Models

**Accelerate your research with `Flash Linear Attention` - a PyTorch and Triton-based library providing efficient implementations for state-of-the-art linear attention models!**  Designed for platform-agnostic performance, this library leverages the power of Triton to deliver optimized kernels across NVIDIA, AMD, and Intel hardware.  [Explore the original repository](https://github.com/fla-org/flash-linear-attention) for more details.

[![hf_model](https://img.shields.io/badge/-Models-gray.svg?logo=huggingface&style=flat-square)](https://huggingface.co/fla-hub)  [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white&style=flat-square)](https://discord.gg/vDaJTmKNcS)

## Key Features:

*   **Platform-Agnostic:** Built entirely with PyTorch and Triton for compatibility across NVIDIA, AMD, and Intel platforms.
*   **Optimized Implementations:** Focus on performance, delivering efficient kernels for a variety of linear attention architectures.
*   **Comprehensive Model Support:**  Includes implementations for RetNet, GLA, Mamba, RWKV, and many more, with more models being added frequently. See the [Models](#models) section for a full list.
*   **Fused Modules:**  Provides pre-built, optimized modules for key operations like rotary embeddings, normalization, and cross-entropy calculations, streamlining model development and training.
*   **Flexible Training Framework:** Integrated with 🔥 `flame`, a minimal and scalable framework for training FLA models.
*   **Easy Integration:** Compatible with Hugging Face Transformers for seamless use within existing workflows.
*   **Zero-Shot Evaluation**: Compatible with  `lm-evaluation-harness` for easy (zero-shot) model evaluations.
*   **Hybrid Model Support:**  Easily combine standard attention layers with linear attention models through flexible configuration.

## Table of Contents

*   [News](#news)
*   [Models](#models)
*   [Installation](#installation)
*   [Usage](#usage)
    *   [Token Mixing](#token-mixing)
    *   [Fused Modules](#fused-modules)
    *   [Generation](#generation)
    *   [Hybrid Models](#hybrid-models)
*   [Training](#training)
*   [Evaluation](#evaluation)
*   [Benchmarks](#benchmarks)
*   [Citation](#citation)
*   [Star History](#star-history)
*   [Acknowledgments](#acknowledgments)

## News

Stay up-to-date with the latest developments:

*   **[2025-09]:** 🐻 Thrilled to announce that [GDN](fla/ops/gated_delta_rule) has been integrated into Qwen3-Next. Check out [the PR](https://github.com/huggingface/transformers/pull/40771) and [their blog post](https://qwenlm.github.io/blog/qwen3_next/) for more infos!
*   **[2025-08]:** 🌲 Added Log-Linear Attention implementation. ([paper](https://arxiv.org/abs/2506.04761)).
*   **[2025-08]:** 🎓 Added MoM implementation. ([paper](https://arxiv.org/abs/2502.13685)).
*   **[2025-07]:** 🐳 Added MLA implementation. ([paper](https://arxiv.org/abs/2405.04434)).
*   **[2025-07]:** 🛣️ Added PaTH Attention. ([paper](https://arxiv.org/abs/2505.16381)).
*   **[2025-06]:** 🎉 Added MesaNet. ([paper](https://arxiv.org/abs/2506.05233)).
*   **[2025-06]:** 🐍 Added Comba implementation. ([paper](https://arxiv.org/abs/2506.02475)).
*   **[2025-05]:** 🎉 Added Rodimus&ast; implementation. ([paper](https://arxiv.org/abs/2410.06577)).
*   **[2025-04]:** 🎉 Added DeltaProduct implementation. ([paper](https://arxiv.org/abs/2502.10297)).
*   **[2025-04]:** 🎉 Added FoX implementation. ([paper](https://arxiv.org/abs/2503.02130)).
*   **[2025-03]:** ~~We have changed the default `initializer_range` to the magic 🐳 0.006~~ The `initializer_range` was rolled back to the default value of 0.02. For actual training, we recommend trying both.
*   **[2025-02]:** 🐳 Added NSA implementations. See kernels [here](fla/ops/nsa).
*   **[2025-01]:** 🔥 We are migrating to `torchtitan`-based training framework. Check out the [flame](https://github.com/fla-org/flame) repo for more details.
*   **[2025-01]:** 🦅 Added RWKV7 implementations (both kernels and models).
*   **[2024-12]:** Integrated `flash-bidirectional-attention`. ([repo](https://github.com/fla-org/flash-bidirectional-linear-attention))
*   **[2024-12]:** 🎉 Added Gated DeltaNet implementation. ([paper](https://arxiv.org/abs/2412.06464)).
*   **[2024-12]:** 🚀 `fla` now officially supports kernels with variable-length inputs.
*   **[2024-11]:** The inputs are now switched from head-first to seq-first format.
*   **[2024-11]:** 💥 `fla` now provides a flexible way for training hybrid models.
*   **[2024-10]:** 🔥 Announcing `flame`, a minimal and scalable framework for training `fla` models. Check out the details [here](training/README.md).
*   **[2024-09]:** `fla` now includes a fused linear and cross-entropy layer, significantly reducing memory usage during training.
*   **[2024-09]:** 🎉 Added GSA implementation. ([paper](https://arxiv.org/abs/2409.07146)).
*   **[2024-05]:** 🎉 Added DeltaNet implementation. ([paper](https://arxiv.org/abs/2102.11174)).
*   **[2024-05]:** 💥 `fla` v0.1: a variety of subquadratic kernels/layers/models integrated (RetNet/GLA/Mamba/HGRN/HGRN2/RWKV6, etc., see [Models](#models)).
*   **[2023-12]:** 💥 Launched `fla`, offering a collection of implementations for state-of-the-art linear attention models.

## Models

A wide range of linear attention models are supported.  The models are roughly sorted by the timeline of their addition to `fla`. We recommend `chunk` mode for training, when available.

| Year | Venue   | Model                | Paper                                                                                                                                         | Code                                                                                            |                                                                                                       |
| :--- | :------ | :------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------: |
| 2023 |         | RetNet               | [Retentive network: a successor to transformer for large language models](https://arxiv.org/abs/2307.08621)                                   | [official](https://github.com/microsoft/torchscale/tree/main)                                   | [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/multiscale_retention.py) |
| 2024 | ICML    | GLA                  | [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)                                      | [official](https://github.com/berlino/gated_linear_attention)                                   |         [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gla.py)          |
| 2024 | ICML    | Based                | [Simple linear attention language models balance the recall-throughput tradeoff](https://arxiv.org/abs/2402.18668)                            | [official](https://github.com/HazyResearch/based)                                               |        [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/based.py)         |
| 2024 | ACL     | Rebased              | [Linear Transformers with Learnable Kernel Functions are Better In-Context Models](https://arxiv.org/abs/2402.10644)                          | [official](https://github.com/corl-team/rebased/)                                               |       [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/rebased.py)        |
| 2024 | NeurIPS | DeltaNet             | [Parallelizing Linear Transformers with Delta Rule  over Sequence Length](https://arxiv.org/abs/2406.06484)                                   | [official](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/delta_net.py) |      [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/delta_net.py)       |
| 2022 | ACL     | ABC                  | [ABC: Attention with Bounded-memory Control](https://arxiv.org/abs/2110.02488)                                                                |                                                                                                 |         [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/abc.py)          |
| 2023 | NeurIPS | HGRN                 | [Hierarchically Gated Recurrent Neural Network for Sequence Modeling](https://openreview.net/forum?id=P1TCHxJwLB)                             | [official](https://github.com/OpenNLPLab/HGRN)                                                  |         [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/hgrn.py)         |
| 2024 | COLM    | HGRN2                | [HGRN2: Gated Linear RNNs with State Expansion](https://arxiv.org/abs/2404.07904)                                                             | [official](https://github.com/OpenNLPLab/HGRN2)                                                 |        [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/hgrn2.py)         |
| 2024 | COLM    | RWKV6                | [Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence](https://arxiv.org/abs/2404.05892)                                    | [official](https://github.com/RWKV/RWKV-LM)                                                     |        [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/rwkv6.py)         |
| 2024 |         | LightNet             | [You Only Scan Once: Efficient Multi-dimension Sequential Modeling with LightNet](https://arxiv.org/abs/2405.21022)                           | [official](https://github.com/OpenNLPLab/LightNet)                                              |       [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/lightnet.py)       |
| 2025 | ICLR    | Samba                | [Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling](https://arxiv.org/abs/2406.07522)                 | [official](https://github.com/microsoft/Samba)                                                  |          [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/samba)          |
| 2024 | ICML    | Mamba2               | [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) | [official](https://github.com/state-spaces/mamba)                                               |         [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/mamba2)          |
| 2024 | NeurIPS | GSA                  | [Gated Slot Attention for Efficient Linear-Time Sequence Modeling](https://arxiv.org/abs/2409.07146)                                          | [official](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models/gsa)          |           [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models/gsa)           |
| 2025 | ICLR    | Gated DeltaNet       | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)                                                    | [official](https://github.com/NVlabs/GatedDeltaNet)                                             |      [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule)      |
| 2025 |         | RWKV7                | [RWKV-7 "Goose" with Expressive Dynamic State Evolution](https://arxiv.org/abs/2503.14456)                                                    | [official](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7)                                |           [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/rwkv7)            |
| 2025 |         | NSA                  | [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089)                         |                                                                                                 |            [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/nsa)             |
| 2025 | ICLR    | FoX                  | [Forgetting Transformer: Softmax Attention with a Forget Gate](https://arxiv.org/abs/2503.02130)                                              | [official](https://github.com/zhixuan-lin/forgetting-transformer)                               |      [fla](https://github.com/flash-linear-attention/tree/main/fla/ops/forgetting_attn)       |
| 2025 |         | DeltaProduct         | [DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](https://arxiv.org/abs/2502.10297)                            |                                                                                                 |  [fla](https://github.com/flash-linear-attention/tree/main/fla/layers/gated_deltaproduct.py)  |
| 2025 | ICLR    | Rodimus&ast;         | [Rodimus*: Breaking the Accuracy-Efficiency Trade-Off with Efficient Attentions](https://arxiv.org/abs/2410.06577)                            | [official](https://github.com/codefuse-ai/rodimus)                                              |       [fla](https://github.com/flash-linear-attention/blob/main/fla/layers/rodimus.py)        |
| 2025 |         | MesaNet              | [MesaNet: Sequence Modeling by Locally Optimal Test-Time Training](https://arxiv.org/abs/2506.05233)                                          |                                                                                                 |       [fla](https://github.com/flash-linear-attention/blob/main/fla/layers/mesa_net.py)       |
| 2025 |         | Comba                | [Comba: Improving Bilinear RNNs with Closed-loop Control](https://arxiv.org/abs/2506.02475)                                                   | [official](https://github.com/AwesomeSeq/Comba-triton)                                          |        [fla](https://github.com/flash-linear-attention/blob/main/fla/layers/comba.py)         |
| 2025 |         | PaTH                 | [PaTH Attention: Position Encoding via Accumulating Householder Transformations](https://arxiv.org/abs/2505.16381)                            |                                                                                                 |      [fla](https://github.com/flash-linear-attention/blob/main/fla/layers/path_attn.py)       |
| 2025 |         | MoM                  | [MoM: Linear Sequence Modeling with Mixture-of-Memories](https://arxiv.org/abs/2502.13685)                                                    | [official](https://github.com/OpenSparseLLMs/MoM)                                               |         [fla](https://github.com/flash-linear-attention/blob/main/fla/layers/mom.py)          |
| 2025 |         | Log-Linear Attention | [Log-Linear Attention](https://arxiv.org/abs/2506.04761)                                                                                      | [official](https://github.com/HanGuo97/log-linear-attention)                                    |      [fla](https://github.com/flash-linear-attention/tree/main/fla/ops/log_linear_attn)       |

## Installation

Ensure you have the following dependencies installed:

*   [PyTorch](https://pytorch.org/) >= 2.5
*   [Triton](https://github.com/openai/triton) >=3.0 (or nightly version, see [FAQs](FAQs.md))
*   [einops](https://einops.rocks/)
*   [transformers](https://github.com/huggingface/transformers) >=4.45.0
*   [datasets](https://github.com/huggingface/datasets) >=3.3.0

Starting from v0.3.2, the packages published on PyPI are `fla-core` and `flash-linear-attention`. The former contains all our customized kernels and only depends on PyTorch, Triton, and einops. The latter is an extension package of the former, containing `fla/layers` and `fla/models`, and depends on transformers. We also provide Triton implementations for conv1d operations, so causal-conv1d is not required.

Install using pip:

```sh
pip install flash-linear-attention
```

For the latest features and updates, install from source. Remember to uninstall both `fla-core` and `flash-linear-attention` first:

```sh
pip uninstall fla-core flash-linear-attention -y && pip install -U git+https://github.com/fla-org/flash-linear-attention
```

or manage `fla` with submodules
```sh
git submodule add https://github.com/fla-org/flash-linear-attention.git 3rdparty/flash-linear-attention
ln -s 3rdparty/flash-linear-attention/fla fla
```

If you have installed `triton-nightly` and `torch` pre version, please use the following command:
```sh
pip install einops ninja datasets transformers numpy
# uninstall both packages first to ensure a successful upgrade
pip uninstall fla-core flash-linear-attention -y && pip install -U --no-use-pep517 git+https://github.com/fla-org/flash-linear-attention --no-deps
```

## Usage

### Token Mixing

Use the `fla.layers` to easily integrate linear attention layers into your models.  Replace standard multihead attention layers with `fla`'s implementations.

```py
>>> import torch
>>> from fla.layers import MultiScaleRetention
>>> batch_size, num_heads, seq_len, hidden_size = 32, 4, 2048, 1024
>>> device, dtype = 'cuda:0', torch.bfloat16
>>> retnet = MultiScaleRetention(hidden_size=hidden_size, num_heads=num_heads).to(device=device, dtype=dtype)
>>> x = torch.randn(batch_size, seq_len, hidden_size).to(device=device, dtype=dtype)
>>> y, *_ = retnet(x)
>>> y.shape
torch.Size([32, 2048, 1024])
```

Here's an example to initialize a GLA model from the default configs in `fla`:

```py
>>> from fla.models import GLAConfig
>>> from transformers import AutoModelForCausalLM
>>> config = GLAConfig()
>>> config
GLAConfig {
  "attn": null,
  "attn_mode": "chunk",
  "bos_token_id": 1,
  "clamp_min": null,
  "conv_size": 4,
  "elementwise_affine": true,
  "eos_token_id": 2,
  "expand_k": 0.5,
  "expand_v": 1,
  "feature_map": null,
  "fuse_cross_entropy": true,
  "fuse_norm": true,
  "fuse_swiglu": true,
  "hidden_act": "swish",
  "hidden_ratio": 4,
  "hidden_size": 2048,
  "initializer_range": 0.006,
  "intermediate_size": null,
  "max_position_embeddings": 2048,
  "model_type": "gla",
  "norm_eps": 1e-06,
  "num_heads": 4,
  "num_hidden_layers": 24,
  "num_kv_heads": null,
  "tie_word_embeddings": false,
  "transformers_version": "4.50.1",
  "use_cache": true,
  "use_gk": true,
  "use_gv": false,
  "use_output_gate": true,
  "use_short_conv": false,
  "vocab_size": 32000
}

>>> AutoModelForCausalLM.from_config(config)
GLAForCausalLM(
  (model): GLAModel(
    (embeddings): Embedding(32000, 2048)
    (layers): ModuleList(
      (0-23): 24 x GLABlock(
        (attn_norm): RMSNorm(2048, eps=1e-06)
        (attn): GatedLinearAttention(
          (q_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (k_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (v_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (g_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (gk_proj): Sequential(
            (0): Linear(in_features=2048, out_features=16, bias=False)
            (1): Linear(in_features=16, out_features=1024, bias=True)
          )
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (g_norm_swish_gate): FusedRMSNormGated(512, eps=1e-06, activation=swish)
        )
        (mlp_norm): RMSNorm(2048, eps=1e-06)
        (mlp): GatedMLP(
          (gate_proj): Linear(in_features=2048, out_features=5632, bias=False)
          (up_proj): Linear(in_features=2048, out_features=5632, bias=False)
          (down_proj): Linear(in_features=5632, out_features=2048, bias=False)
          (swiglu_linear): SwiGLULinear()
        )
      )
    )
    (norm): RMSNorm(2048, eps=1e-06)
  )
  (lm_head): Linear(in_features=2048, out_features=32000, bias=False)
)
```

### Fused Modules

Utilize `fla.modules` for optimized building blocks:

*   [`Rotary Embedding`](fla/modules/rotary.py): rotary positional embeddings.
*   [`Norm Layers`](fla/modules/layernorm.py): `RMSNorm`, `LayerNorm`, `GroupNorm` with fused linear layers.
*   [`Norm Layers with Gating`](fla/modules/fused_norm_gate.py): Norm layers combined with sigmoid or swish gating.
*   [`Cross Entropy`](fla/modules/fused_cross_entropy.py): Fast Triton-based cross entropy loss.
*   [`Linear Cross Entropy`](fla/modules/fused_linear_cross_entropy.py): Fused linear layer and cross entropy loss.
*   [`Linear KL Divergence`](fla/modules/fused_kl_div.py): Fused linear layer and KL divergence loss.

> [!IMPORTANT]
> Enable/disable the `fuse_linear_cross_entropy` in the model configuration.
>
>  This implementation is memory-efficient but can impact numerical precision, so it is disabled by default.
>  If training becomes unstable, disable this feature.

### Generation

Generate text using the 🤗 text generation APIs after pretraining.
```py
>>> import fla
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> name = 'fla-hub/gla-1.3B-100B'
>>> tokenizer = AutoTokenizer.from_pretrained(name)
>>> model = AutoModelForCausalLM.from_pretrained(name).cuda()
>>> input_prompt = "Power goes with permanence. Impermanence is impotence. And rotation is castration."
>>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.cuda()
>>> outputs = model.generate(input_ids, max_length=64)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
```

Find pretrained models on [`fla-hub`](https://huggingface.co/fla-hub).
```py
>>> from huggingface_hub import list_models
>>> for model in list_models(author='fla-hub'): print(model.id)
```

### Hybrid Models

Easily combine standard and linear attention layers.

```py
>>> from fla.models import SambaConfig
>>> from transformers import AutoModelForCausalLM
>>> config = SambaConfig(num_hidden_layers=2)
>>> config.attn = {
  'layers': [1],
  'num_heads': 18,
  'num_kv_heads': 18,
  'qkv_bias': False,
  'rope_theta': 10000.,
  'window_size': 2048
}
>>> AutoModelForCausalLM.from_config(config)
SambaForCausalLM(
  (backbone): SambaModel(
    (embeddings): Embedding(32000, 2304)
    (layers): ModuleList(
      (0): SambaBlock(
        (mixer_norm): RMSNorm(2304, eps=1e-05)
        (mixer): Mamba(
          (conv1d): Conv1d(4608, 4608, kernel_size=(4,), stride=(1,), padding=(3,), groups=4608)
          (in_proj): Linear(in_features=2304, out_features=9216, bias=False)
          (x_proj): Linear(in_features=4608, out_features=176, bias=False)
          (dt_proj): Linear(in_features=144, out_features=4608, bias=True)
          (out_proj): Linear(in_features=4608, out_features=2304, bias=False)
        )
        (mlp_norm): RMSNorm(2304, eps=1e-05)
        (mlp): GatedMLP(
          (gate_proj): Linear(in_features=2304, out_features=6144, bias=False)
          (up_proj): Linear(in_features=2304, out_features=6144, bias=False)
          (down_proj): Linear(in_features=6144, out_features=2304, bias=False)
          (swiglu_linear): SwiGLULinear()
        )
      )
      (1): SambaBlock(
        (mixer_norm): RMSNorm(2304, eps=1e-05)
        (mixer): Attention(
          (q_proj): Linear(in_features=2304, out_features=2304, bias=False)
          (k_proj): Linear(in_features=2304, out_features=2304, bias=False)
          (v_proj): Linear(in_features=2304, out_features=2304, bias=False)
          (o_proj): Linear(in_features=2304, out_features=2304, bias=False)
          (rotary): RotaryEmbedding(dim=128, base=10000.0, interleaved=False, pos_idx_in_fp32=True)
        )
        (mlp_norm): RMSNorm(2304, eps=1e-05)
        (mlp): GatedMLP(
          (gate_proj): Linear(in_features=2304, out_features=6144, bias=False)
          (up_proj): Linear(in_features=2304, out_features=6144, bias=False)
          (down_proj): Linear(in_features=6144, out_features=2304, bias=False)
          (swiglu_linear): SwiGLULinear()
        )
      )
    )
    (norm_f): RMSNorm(2304, eps=1e-05)
  )
  (lm_head): Linear(in_features=2304, out_features=32000, bias=False)
)
```

During inference, you **DO NOT** need to revise anything for generation!
The model will produce output as-is, without any need for additional configurations or modifications.

## Training

Use `flame` for efficient training: [🔥 `flame`](https://github.com/fla-org/flame).

See [the GLA example](https://github.com/fla-org/flash-linear-attention/blob/main/examples/training.md) for details.

## Evaluation

Evaluate models using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) library.

1.  Install `lm_eval` following [their instructions](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md).

2.  Run evaluation:

```sh
$ MODEL='fla-hub/gla-1.3B-100B'
$ python -m evals.harness --model hf \
    --model_args pretrained=$MODEL,dtype=bfloat16 \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
    --batch_size 64 \
    --num_fewshot 0 \
    --device cuda \
    --show_config
```

3.  Multi-GPU Evaluation with Hugging Face accelerate 🚀

```sh
$ MODEL='fla-hub/gla-1.3B-100B'
$ accelerate launch -m evals.harness --model hf  \
    --model_args pretrained=$MODEL,dtype=bfloat16,trust_remote_code=True  \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
    --batch_size 64  \
    --num_fewshot 0  \
    --device cuda  \
    --show_config  \
    --trust_remote_code
```

4.  📏 RULER Benchmark suite

RULER benchmarks can be used to test long-context tasks.  Ensure you have the latest version of `lm-evaluation-harness`.

```
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
pip install lm_eval["ruler"]
```

Run evaluation:

```sh
$ accelerate launch -m evals.harness \
    --output_path $OUTPUT \
    --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,