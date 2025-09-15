<div align="center">

# Flash Linear Attention: Accelerating Transformer Models with Efficient Implementations

[![Hugging Face Models](https://img.shields.io/badge/-Models-gray.svg?logo=huggingface&style=flat-square)](https://huggingface.co/fla-hub)  [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white&style=flat-square)](https://discord.gg/vDaJTmKNcS)

**Get blazing-fast performance for your transformer models!** Flash Linear Attention provides a collection of efficient, Triton-based implementations of state-of-the-art linear attention mechanisms, designed for platform-agnostic performance. Check out the original repo [here](https://github.com/fla-org/flash-linear-attention)!

</div>

**Key Features:**

*   **Platform-Agnostic**:  Pure PyTorch and Triton implementations, verified on NVIDIA, AMD, and Intel platforms.
*   **Extensive Model Support**: Implementations for RetNet, GLA, Mamba, RWKV, GSA, DeltaNet, and more (see [Models](#models)).
*   **Fused Modules for Efficiency**:  Includes fused modules for rotary embeddings, normalization, and cross-entropy loss, optimizing memory usage and training speed.
*   **Hybrid Model Flexibility**: Seamlessly integrate standard attention layers into linear attention models.
*   **Easy Integration**:  Compatible with the Hugging Face Transformers library for straightforward model use and evaluation.
*   **Comprehensive Benchmarks**: Performance comparisons, including RetNet vs. FlashAttention2, demonstrating significant speedups.
*   **Active Development**: Continuously updated with the latest research and features.

**Jump to Sections:**

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

A curated list of recent updates and exciting developments in the Flash Linear Attention library.

*   **[2025-09]:** 🐻 GDN ([fla/ops/gated\_delta\_rule](fla/ops/gated_delta_rule)) has been integrated into Qwen3-Next.
*   **[2025-08]:** 🌲 Added Log-Linear Attention implementation ([paper](https://arxiv.org/abs/2506.04761)).
*   **[2025-08]:** 🎓 Added MoM implementation ([paper](https://arxiv.org/abs/2502.13685)).
*   **[2025-07]:** 🐳 Added MLA implementation ([paper](https://arxiv.org/abs/2405.04434)).
*   **[2025-07]:** 🛣️ Added PaTH Attention ([paper](https://arxiv.org/abs/2505.16381)).
*   **[2025-06]:** 🎉 Added MesaNet ([paper](https://arxiv.org/abs/2506.05233)).
*   **[2025-06]:** 🐍 Added Comba implementation ([paper](https://arxiv.org/abs/2506.02475)).
*   **[2025-05]:** 🎉 Added Rodimus&ast; implementation ([paper](https://arxiv.org/abs/2410.06577)).
*   **[2025-04]:** 🎉 Added DeltaProduct implementation ([paper](https://arxiv.org/abs/2502.10297)).
*   **[2025-04]:** 🎉 Added FoX implementation ([paper](https://arxiv.org/abs/2503.02130)).
*   **[2025-03]:** ~~We have changed the default `initializer_range` to the magic 🐳 0.006~~ The `initializer_range` was rolled back to the default value of 0.02. For actual training, we recommend trying both.
*   **[2025-02]:** 🐳 Added NSA implementations (see kernels [here](fla/ops/nsa)).
*   **[2025-01]:** 🔥 Migrating to `torchtitan`-based training framework. Check out the [flame](https://github.com/fla-org/flame) repo.
*   **[2025-01]:** 🦅 Added RWKV7 implementations.
*   **[2024-12]:** Integrated `flash-bidirectional-attention` to `fla-org` ([repo](https://github.com/fla-org/flash-bidirectional-linear-attention))
*   **[2024-12]:** 🎉 Added Gated DeltaNet implementation ([paper](https://arxiv.org/abs/2412.06464)).
*   **[2024-12]:** 🚀 `fla` officially supports kernels with variable-length inputs.
*   **[2024-11]:** Inputs are now switched from head-first to seq-first format.
*   **[2024-11]:** 💥 `fla` provides flexible training for hybrid models.
*   **[2024-10]:** 🔥 Announcing `flame`, a scalable framework for training `fla` models. Check out the details [here](training/README.md).
*   **[2024-09]:** Fused linear and cross-entropy layer.
*   **[2024-09]:** 🎉 Added GSA implementation ([paper](https://arxiv.org/abs/2409.07146)).
*   **[2024-05]:** 🎉 Added DeltaNet implementation ([paper](https://arxiv.org/abs/2102.11174)).
*   **[2024-05]:** 💥 `fla` v0.1: integrated subquadratic kernels/layers/models (RetNet/GLA/Mamba/HGRN/HGRN2/RWKV6, etc.).
*   **[2023-12]:** 💥 Launched `fla`, offering a collection of implementations for state-of-the-art linear attention models.

## Models

A comprehensive table showcasing the various linear attention models supported by Flash Linear Attention, along with their corresponding publications and code locations.  The recommended training mode is `chunk` when available.

| Year | Venue   | Model                | Paper                                                                                                                                         | Code                                                                                            |                                                                                                       |
| :--- | :------ | :------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------: |
| 2023 |         | RetNet               | [Retentive network: a successor to transformer for large language models](https://arxiv.org/abs/2307.08621)                                   | [official](https://github.com/microsoft/torchscale/tree/main)                                   | [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/multiscale_retention.py) |
| 2024 | ICML    | GLA                  | [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)                                      | [official](https://github.com/berlino/gated_linear_attention)                                   |         [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gla.py)          |
| 2024 | ICML    | Based                | [Simple linear attention language models balance the recall-throughput tradeoff](https://arxiv.org/abs/2402.18668)                            | [official](https://github.com/HazyResearch/based)                                               |        [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/based.py)         |
| 2024 | ACL     | Rebased              | [Linear Transformers with Learnable Kernel Functions are Better In-Context Models](https://arxiv.org/abs/2402.10644)                          | [official](https://github.com/corl-team/rebased/)                                               |       [fla](https://github.com/flash-linear-attention/blob/main/fla/layers/rebased.py)        |
| 2024 | NeurIPS | DeltaNet             | [Parallelizing Linear Transformers with Delta Rule  over Sequence Length](https://arxiv.org/abs/2406.06484)                                   | [official](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/delta_net.py) |      [fla](https://github.com/flash-linear-attention/blob/main/fla/layers/delta_net.py)       |
| 2022 | ACL     | ABC                  | [ABC: Attention with Bounded-memory Control](https://arxiv.org/abs/2110.02488)                                                                |                                                                                                 |         [fla](https://github.com/flash-linear-attention/blob/main/fla/layers/abc.py)          |
| 2023 | NeurIPS | HGRN                 | [Hierarchically Gated Recurrent Neural Network for Sequence Modeling](https://openreview.net/forum?id=P1TCHxJwLB)                             | [official](https://github.com/OpenNLPLab/HGRN)                                                  |         [fla](https://github.com/flash-linear-attention/blob/main/fla/layers/hgrn.py)         |
| 2024 | COLM    | HGRN2                | [HGRN2: Gated Linear RNNs with State Expansion](https://arxiv.org/abs/2404.07904)                                                             | [official](https://github.com/OpenNLPLab/HGRN2)                                                 |        [fla](https://github.com/flash-linear-attention/blob/main/fla/layers/hgrn2.py)         |
| 2024 | COLM    | RWKV6                | [Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence](https://arxiv.org/abs/2404.05892)                                    | [official](https://github.com/RWKV/RWKV-LM)                                                     |        [fla](https://github.com/flash-linear-attention/blob/main/fla/layers/rwkv6.py)         |
| 2024 |         | LightNet             | [You Only Scan Once: Efficient Multi-dimension Sequential Modeling with LightNet](https://arxiv.org/abs/2405.21022)                           | [official](https://github.com/OpenNLPLab/LightNet)                                              |       [fla](https://github.com/flash-linear-attention/blob/main/fla/layers/lightnet.py)       |
| 2025 | ICLR    | Samba                | [Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling](https://arxiv.org/abs/2406.07522)                 | [official](https://github.com/microsoft/Samba)                                                  |          [fla](https://github.com/flash-linear-attention/blob/main/fla/models/samba)          |
| 2024 | ICML    | Mamba2               | [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) | [official](https://github.com/state-spaces/mamba)                                               |         [fla](https://github.com/flash-linear-attention/blob/main/fla/models/mamba2)          |
| 2024 | NeurIPS | GSA                  | [Gated Slot Attention for Efficient Linear-Time Sequence Modeling](https://arxiv.org/abs/2409.07146)                                          | [official](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models/gsa)          |           [fla](https://github.com/flash-linear-attention/tree/main/fla/models/gsa)           |
| 2025 | ICLR    | Gated DeltaNet       | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)                                                    | [official](https://github.com/NVlabs/GatedDeltaNet)                                             |      [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule)      |
| 2025 |         | RWKV7                | [RWKV-7 "Goose" with Expressive Dynamic State Evolution](https://arxiv.org/abs/2503.14456)                                                    | [official](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7)                                |           [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/rwkv7)            |
| 2025 |         | NSA                  | [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089)                         |                                                                                                 |            [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/nsa)             |
| 2025 | ICLR    | FoX                  | [Forgetting Transformer: Softmax Attention with a Forget Gate](https://arxiv.org/abs/2503.02130)                                              | [official](https://github.com/zhixuan-lin/forgetting-transformer)                               |      [fla](https://github.com/flash-linear-attention/blob/main/fla/ops/forgetting_attn)       |
| 2025 |         | DeltaProduct         | [DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](https://arxiv.org/abs/2502.10297)                            |                                                                                                 |  [fla](https://github.com/flash-linear-attention/tree/main/fla/layers/gated_deltaproduct.py)  |
| 2025 | ICLR    | Rodimus&ast;         | [Rodimus*: Breaking the Accuracy-Efficiency Trade-Off with Efficient Attentions](https://arxiv.org/abs/2410.06577)                            | [official](https://github.com/codefuse-ai/rodimus)                                              |       [fla](https://github.com/flash-linear-attention/blob/main/fla/layers/rodimus.py)        |
| 2025 |         | MesaNet              | [MesaNet: Sequence Modeling by Locally Optimal Test-Time Training](https://arxiv.org/abs/2506.05233)                                          |                                                                                                 |       [fla](https://github.com/flash-linear-attention/blob/main/fla/layers/mesa_net.py)       |
| 2025 |         | Comba                | [Comba: Improving Bilinear RNNs with Closed-loop Control](https://arxiv.org/abs/2506.02475)                                                   | [official](https://github.com/AwesomeSeq/Comba-triton)                                          |        [fla](https://github.com/flash-linear-attention/blob/main/fla/layers/comba.py)         |
| 2025 |         | PaTH                 | [PaTH Attention: Position Encoding via Accumulating Householder Transformations](https://arxiv.org/abs/2505.16381)                            |                                                                                                 |      [fla](https://github.com/flash-linear-attention/blob/main/fla/layers/path_attn.py)       |
| 2025 |         | MoM                  | [MoM: Linear Sequence Modeling with Mixture-of-Memories](https://arxiv.org/abs/2502.13685)                                                    | [official](https://github.com/OpenSparseLLMs/MoM)                                               |         [fla](https://github.com/flash-linear-attention/blob/main/fla/layers/mom.py)          |
| 2025 |         | Log-Linear Attention | [Log-Linear Attention](https://arxiv.org/abs/2506.04761)                                                                                      | [official](https://github.com/HanGuo97/log-linear-attention)                                    |      [fla](https://github.com/flash-linear-attention/tree/main/fla/ops/log_linear_attn)       |

## Installation

Follow these steps to quickly set up Flash Linear Attention.

*   **Prerequisites:**
    *   [PyTorch](https://pytorch.org/) >= 2.5
    *   [Triton](https://github.com/openai/triton) >=3.0 (or nightly version, see [FAQs](FAQs.md))
    *   [einops](https://einops.rocks/)
    *   [transformers](https://github.com/huggingface/transformers) >=4.45.0
    *   [datasets](https://github.com/huggingface/datasets) >=3.3.0

*   **Install with pip:**
    ```bash
    pip install flash-linear-attention
    ```

*   **Install from Source for Latest Features:**
    ```bash
    # uninstall both packages first to ensure a successful upgrade
    pip uninstall fla-core flash-linear-attention -y && pip install -U git+https://github.com/fla-org/flash-linear-attention
    ```
    or manage `fla` with submodules
    ```sh
    git submodule add https://github.com/fla-org/flash-linear-attention.git 3rdparty/flash-linear-attention
    ln -s 3rdparty/flash-linear-attention/fla fla
    ```

    *If you have installed `triton-nightly` and `torch` pre version:*
     ```bash
     pip install einops ninja datasets transformers numpy
     # uninstall both packages first to ensure a successful upgrade
     pip uninstall fla-core flash-linear-attention -y && pip install -U --no-use-pep517 git+https://github.com/fla-org/flash-linear-attention --no-deps
     ```

*   **CI Status:**  Installation is verified on multiple platforms.

    [![nvidia-4090-ci](https://github.com/fla-org/flash-linear-attention/actions/workflows/nvidia-4090.yml/badge.svg?branch=main&event=push)](https://github.com/fla-org/flash-linear-attention/actions/workflows/nvidia-4090.yml) [![nvidia-a100-ci](https://github.com/fla-org/flash-linear-attention/actions/workflows/nvidia-a100.yml/badge.svg?branch=main)](https://github.com/fla-org/flash-linear-attention/actions/workflows/nvidia-a100.yml) [![nvidia-h100-ci](https://github.com/fla-org/flash-linear-attention/actions/workflows/nvidia-h100.yml/badge.svg?branch=main&event=push)](https://github.com/fla-org/flash-linear-attention/actions/workflows/nvidia-h100.yml) [![intel-b580-ci](https://github.com/fla-org/flash-linear-attention/actions/workflows/intel-b580.yml/badge.svg?event=push)](https://github.com/fla-org/flash-linear-attention/actions/workflows/intel-b580.yml)

## Usage

Quickly integrate Flash Linear Attention into your projects.

### Token Mixing

Replace standard attention layers with optimized linear attention layers from `fla.layers`.

```python
import torch
from fla.layers import MultiScaleRetention

batch_size, num_heads, seq_len, hidden_size = 32, 4, 2048, 1024
device, dtype = 'cuda:0', torch.bfloat16
retnet = MultiScaleRetention(hidden_size=hidden_size, num_heads=num_heads).to(device=device, dtype=dtype)

x = torch.randn(batch_size, seq_len, hidden_size).to(device=device, dtype=dtype)
y, *_ = retnet(x)

print(y.shape)  # Output: torch.Size([32, 2048, 1024])
```

### Fused Modules

Leverage fused modules in `fla.modules` for enhanced speed and efficiency.

*   `Rotary Embedding`
*   `RMSNorm`, `LayerNorm`, `GroupNorm`, and their fused counterparts for linear layers (`RMSNormLinear`, etc.).
*   `Norm Layers with Gating`
*   `Fused Cross Entropy`
*   `Fused Linear Cross Entropy`
*   `Fused Linear KL Divergence`

> [!IMPORTANT]
> Control `fuse_linear_cross_entropy` in the model configuration to enable/disable the fused linear cross entropy loss.

### Generation

Generate text with pretrained models.

```python
import fla
from transformers import AutoModelForCausalLM, AutoTokenizer

name = 'fla-hub/gla-1.3B-100B'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).cuda()
input_prompt = "Power goes with permanence. Impermanence is impotence. And rotation is castration."
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.cuda()
outputs = model.generate(input_ids, max_length=64)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

Find pre-trained models on [fla-hub](https://huggingface.co/fla-hub).

```python
from huggingface_hub import list_models
for model in list_models(author='fla-hub'): print(model.id)
```

### Hybrid Models

Combine standard and linear attention layers.  Customize model behavior through configuration parameters.

```python
from fla.models import SambaConfig
from transformers import AutoModelForCausalLM

config = SambaConfig(num_hidden_layers=2)
config.attn = {
  'layers': [1],
  'num_heads': 18,
  'num_kv_heads': 18,
  'qkv_bias': False,
  'rope_theta': 10000.,
  'window_size': 2048
}
model = AutoModelForCausalLM.from_config(config)
```

During inference, use the model without any modifications!

## Training

Train `fla` models efficiently with the [🔥 `flame`](https://github.com/fla-org/flame) framework, built on `torchtitan`.  Check out the [GLA example](https://github.com/fla-org/flash-linear-attention/blob/main/examples/training.md) for details.

## Evaluation

Evaluate models using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) library.

1.  Install `lm_eval`.

2.  Run evaluation (example):
    ```bash
    MODEL='fla-hub/gla-1.3B-100B'
    python -m evals.harness --model hf \
        --model_args pretrained=$MODEL,dtype=bfloat16 \
        --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
        --batch_size 64 \
        --num_fewshot 0 \
        --device cuda \
        --show_config
    ```

3.  Multi-GPU evaluation with Hugging Face Accelerate:

    ```bash
    MODEL='fla-hub/gla-1.3B-100B'
    accelerate launch -m evals.harness --model hf  \
        --model_args pretrained=$MODEL,dtype=bfloat16,trust_remote_code=True  \
        --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
        --batch_size 64  \
        --num_fewshot 0  \
        --device cuda  \
        --show_config  \
        --trust_remote_code
    ```

4.  Evaluate on RULER benchmarks:

    ```bash
    accelerate launch -m evals.harness \
        --output_path $OUTPUT \
        --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_hotpot,ruler_qa_squad \
        --model_args pretrained=$MODEL,dtype=bfloat16,max_length=32768,trust_remote_code=True \
        --metadata='{"max_seq_lengths":[4096,8192,16384,32768]}' \
        --batch_size 2 \
        --show_config  \
        --trust_remote_code
    ```

> [!Tip]
> If tasks are missing in `lm-evaluation-harness`, run `from lm_eval.tasks import TaskManager; TaskManager().initialize_tasks()` before `lm_eval.evaluate()` or `lm_eval.simple_evaluate()`.

## Benchmarks

Performance comparisons of our Triton-based RetNet implementation with CUDA-based FlashAttention2 on an H100 80GB GPU.

```py
# example of `python benchmark_retention.py` output
Performance:
         T  chunk_fwd  parallel_fwd  flash_fwd  chunk_fwdbwd  parallel_fwdbwd  flash_fwdbwd
0    128.0   0.264032      0.243536   0.083488      1.301856         1.166784      0.320704
1    256.0   0.273472      0.252848   0.094304      1.345872         1.300608      0.807936
2    512.0   0.303600      0.278896   0.098112      1.503168         1.433184      0.857216
3   1024.0   0.357248      0.367360   0.156528      1.773552         2.303424      1.160864
4   2048.0   0.454624      0.605616   0.340928      2.283728         4.483360      1.955936
5   4096.0   0.638960      1.378016   1.004992      3.374720        12.271215      4.813776
6   8192.0   1.012352      4.201344   3.625008      5.581808        40.833618     15.023697
7  16384.0   1.748512     14.489664  13.710080     10.191552       153.093765     54.336864
```

<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/c2607015-63af-43d1-90d1-ad5fe1670a03">
</div>

## Citation

If you utilize this repository, please cite our work:

```bib
@software{yang2024fla,
  title  = {FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism},
  author = {Yang, Songlin and Zhang, Yu},
  url    = {https://github.com/fla-org/flash-linear-attention},
  month  = jan,
  year   = {2024}
}
@inproceedings{yang2024gdn,
  title     = {Gated Delta Networks: Improving Mamba2 with Delta Rule},
  author    = {Songlin Yang and Jan Kautz and Ali Hatamizadeh},
  booktitle = {Proceedings of ICLR},
  year      = {2025}
}
@inproceedings{yang2024deltanet,
  title     = {Parallelizing Linear Transformers with the Delta Rule over Sequence Length},
  author    = {Yang, Songlin and Wang, Bailin and Zhang, Yu and Shen, Yikang and Kim, Yoon},
  booktitle = {Proceedings of NeurIPS},
  year      = {2024}
}
@inproceedings{zhang2024gsa,
  title     = {Gated Slot Attention for Efficient Linear-Time Sequence Modeling},
  author    = {Zhang, Yu and Yang, Songlin and Zhu, Ruijie and Zhang, Yue and Cui, Leyang and Wang, Yiqiao and Wang, Bolun and Shi, Freda and Wang, Bailin and Bi, Wei and Zhou, Peng and Fu, Guohong},
  booktitle = {Proceedings of NeurIPS},
  year      = {2024}
}
@inproceedings{qin2024hgrn2,
  title     = {HGRN2: Gated Linear RNNs with State Expansion},
  author    = {Qin, Zhen and Yang, Songlin and Sun, Weixuan and Shen, Xuyang and Li, Dong and Sun, Weigao and Zhong, Yiran},
  booktitle = {Proceedings of COLM},
  year      = {2024}
}
@inproceedings{yang2024gla,
  title     = {Gated Linear Attention Transformers with Hardware-Efficient Training},
  author    = {Yang, Songlin and Wang, Bailin and Shen, Yikang and Panda, Rameswar and Kim, Yoon},
  booktitle = {Proceedings of ICML},
  year      = {2024}
}
```

## Star History

[![Stargazers repo roster for @fla-org/flash-linear-attention](https://bytecrank.com/nastyox/reporoster/php/stargazersSVG.php?user=fla-org&repo=flash-linear-attention)](https://github.com/fla-org/flash-linear-attention/stargazers)

[![Star History Chart](https://api.star-history.com/svg?repos=fla-org/flash-linear-attention&type=Date)](https://star-history.com/#fla-org/flash-linear-attention&Date)

## Acknowledgments

We extend our gratitude to [Bitdeer](https://www.bitdeer.com/) for providing CI server resources.