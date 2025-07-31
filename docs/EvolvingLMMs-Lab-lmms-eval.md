<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png" alt="LMMs-Eval Logo">
</p>

# LMMs-Eval: Your Comprehensive Suite for Evaluating Large Multimodal Models

**[LMMs-Eval on GitHub](https://github.com/EvolvingLMMs-Lab/lmms-eval)** | [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

> Evaluate and benchmark your Large Multimodal Models (LMMs) efficiently with LMMs-Eval, supporting a wide array of tasks across text, image, video, and audio modalities.

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

---

## Key Features:

*   **Broad Task Coverage:** Supports over 100 tasks across text, image, video, and audio, including new benchmarks like [PhyX](https://phyx-bench.github.io/) and [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA).
*   **Extensive Model Support:** Compatible with 30+ LMMs, with ongoing additions of new models like Aero-1-Audio.
*   **Flexible Evaluation:** Includes support for OpenAI API-compatible models, vLLM integration for accelerated evaluation, and SGlang runtime API for faster inference.
*   **Reproducibility:** Offers scripts and environment information to reproduce paper results for LLaVA-1.5.
*   **Comprehensive Results:** Provides detailed evaluation results and access to raw data for the LLaVA series models on various datasets.
*   **Ease of Use:** Simple installation and clear usage examples for various models and evaluation scenarios.
*   **Customization:**  Detailed documentation for adding custom models and datasets.
*   **Active Development:** Benefit from frequent updates, new tasks, and model support, driven by an active community.

## Announcements:

*   **[2025-07]** 🚀🚀 Released `lmms-eval-0.4` with major updates and improvements (see the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md)).  For `lmms-eval-0.3` users, see the `stable/v0d3` branch.
*   **[2025-04]** 🚀🚀 Introduced Aero-1-Audio, with support for batched evaluations.
*   **[2025-07]** 🎉🎉 Added support for the PhyX benchmark.
*   **[2025-06]** 🎉🎉 Added support for the VideoMathQA benchmark.
*   **[2025-02]** 🚀🚀 Integrated `vllm` and `openai_compatible`.

<details>
<summary>Recent Updates (Full Chronological List)</summary>
  - [2025-01] 🎓🎓 Released the new benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826).
  - [2024-12] 🎉🎉 Presented [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296).
  - [2024-11] 🔈🔊 Upgraded `lmms-eval/v0.3.0` to support audio evaluations for audio models.
  - [2024-10] 🎉🎉 Added support for NaturalBench and TemporalBench.
  - [2024-10] 🎉🎉 Added support for VDC, MovieChat-1K, Vinoground, AuroraCap, and MovieChat.
  - [2024-09] 🎉🎉 Added support for MMSearch and MME-RealWorld.
  - [2024-09] ⚙️️⚙️️️️ Upgraded `lmms-eval` to `0.2.3` with more tasks and features.
  - [2024-08] 🎉🎉 Added support for LLaVA-OneVision, Mantis, MVBench, LongVideoBench, and MMStar.
  - [2024-07] 👨‍💻👨‍💻 Upgraded `lmms-eval/v0.2.1` to support more models and tasks.
  - [2024-07] 🎉🎉 Released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench).
  - [2024-06] 🎬🎬 Upgraded `lmms-eval/v0.2.0` to support video evaluations.
  - [2024-03] 📝📝 Released the first version of `lmms-eval`.
</details>

---

## Installation:

**Using `uv` (Recommended):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

**Development Installation:**

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv venv dev
source dev/bin/activate
uv pip install -e .
```

**Optional Dependencies:**

*   For caption datasets (coco, refcoco, nocaps), install Java 1.8:  `conda install openjdk=8`
*   If you want to test [VILA](https://github.com/NVlabs/VILA), you should install the following dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

## Usages:

**Quick Start Examples:**

*   **OpenAI-Compatible Model:**  `bash examples/models/openai_compatible.sh` , `bash examples/models/xai_grok.sh`
*   **vLLM:**  `bash examples/models/vllm_qwen2vl.sh`
*   **LLaVA-OneVision:**  `bash examples/models/llava_onevision.sh`
*   **LLaMA-3.2-Vision:**  `bash examples/models/llama_vision.sh`
*   **Qwen2-VL:**  `bash examples/models/qwen2_vl.sh` , `bash examples/models/qwen2_5_vl.sh`
*   **LLaVA (on MME):** (Requires LLaVA repo) `bash examples/models/llava_next.sh`
*   **Tensor Parallel (LLaVA-Next-72b):** `bash examples/models/tensor_parallel.sh`
*   **SGLang (LLaVA-Next-72b):** `bash examples/models/sglang.sh`
*   **vLLM (LLaVA-Next-72b):** `bash examples/models/vllm_qwen2vl.sh`
*   **More Parameters:**  `python3 -m lmms_eval --help`

**Environmental Variables:**

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

**Common Issues & Solutions:**

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26;
python3 -m pip install sentencepiece;
```

## Adding Customized Model and Dataset

Refer to the [documentation](docs/README.md).

---

## Acknowledgement

LMMs-Eval is a fork of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Citations

```shell
@misc{zhang2024lmmsevalrealitycheckevaluation,
      title={LMMs-Eval: Reality Check on the Evaluation of Large Multimodal Models}, 
      author={Kaichen Zhang and Bo Li and Peiyuan Zhang and Fanyi Pu and Joshua Adrian Cahyono and Kairui Hu and Shuai Liu and Yuanhan Zhang and Jingkang Yang and Chunyuan Li and Ziwei Liu},
      year={2024},
      eprint={2407.12772},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.12772}, 
}

@misc{lmms_eval2024,
    title={LMMs-Eval: Accelerating the Development of Large Multimoal Models},
    url={https://github.com/EvolvingLMMs-Lab/lmms-eval},
    author={Bo Li*, Peiyuan Zhang*, Kaichen Zhang*, Fanyi Pu*, Xinrun Du, Yuhao Dong, Haotian Liu, Yuanhan Zhang, Ge Zhang, Chunyuan Li and Ziwei Liu},
    publisher    = {Zenodo},
    version      = {v0.1.0},
    month={March},
    year={2024}
}