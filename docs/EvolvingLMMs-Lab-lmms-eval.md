<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png" alt="lmms-eval Logo">
</p>

# lmms-eval: The Comprehensive Evaluation Suite for Large Multimodal Models

**Evaluate and advance your Large Multimodal Models (LMMs) with `lmms-eval`, a powerful and versatile evaluation framework.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> **[Original Repository](https://github.com/EvolvingLMMs-Lab/lmms-eval)**

**Key Features:**

*   **Extensive Task Support:** Evaluate LMMs across a wide range of modalities, including text, image, video, and audio with **100+ tasks**  [Supported Tasks](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md).
*   **Broad Model Compatibility:** Supports **30+ models**, including LLaVA, Qwen-VL, Gemini, and many more [Supported Models](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models).
*   **Accelerated Evaluation:**  Integrates `vllm` and `openai_compatible` for faster and more efficient evaluation, as well as support of SGLang.
*   **Reproducibility Focus:**  Provides scripts and resources for reproducing paper results.
*   **Regular Updates:**  Continuously updated with new tasks, models, and features.

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## What's New

*   **[2025-07]** `lmms-eval-0.4` released with significant updates and improvements. See [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-07]** Support for [PhyX](https://phyx-bench.github.io/), a physics-grounded reasoning benchmark.
*   **[2025-06]** Integration of [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA)
*   **[2025-04]** Introduction of [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/), an audio model with batched evaluation support.
*   **[2025-02]** Enhanced with `vllm` and `openai_compatible` for faster evaluation and broader model support.

<details>
<summary>Recent Updates (See more in the original README)</summary>

-   [2025-01] Video-MMMU Benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826).
-   [2024-12] MME-Survey: [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296).
-   [2024-11] Audio Evaluations: Audio models such as Qwen2-Audio and Gemini-Audio.
-   [2024-10] New Tasks: NaturalBench, TemporalBench, VDC, MovieChat-1K, and Vinoground.
-   [2024-09] New Tasks: MMSearch and MME-RealWorld, with inference acceleration.
-   [2024-09]  lmms-eval updated to `0.2.3` with more tasks and features.
-   [2024-08] New Models: LLaVA-OneVision, Mantis, and new tasks.
-   [2024-07] lmms-eval `v0.2.1` supports more models and evaluation tasks.
-   [2024-07] Technical Report and LiveBench Released: [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)!
-   [2024-06] Video Evaluations: video models.
-   [2024-03] First Version of lmms-eval Release

</details>

## Installation

### Using uv (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv sync  # This creates/updates your environment from uv.lock
uv run python -m lmms_eval --help  # Run any command with uv run
uv add <package>  # Updates both pyproject.toml and uv.lock
```

### Alternative Installation

```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

<details>
<summary>Reproducing LLaVA-1.5 Results</summary>

Check [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) to reproduce LLaVA-1.5's paper results. See [results check](miscs/llava_result_check.md) for variations.

</details>

If you plan to use datasets such as `coco`, `refcoco`, and `nocaps`, install:

```bash
conda install openjdk=8
```

## Detailed Evaluation Results

<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>
<br>

Detailed information regarding the datasets included in lmms-eval, including specific details about them, can be found in this section.
  Detailed results are available on a Google Sheet [here](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing).
  Raw data exported from Weights & Biases is available [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

<p align="center" width="100%">
<img src="https://i.postimg.cc/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%">
</p>

</details>

If you want to test [VILA](https://github.com/NVlabs/VILA), install the following:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

> **Contribute!** Provide feedback and suggest improvements via issues or pull requests on GitHub.

## Getting Started

> More examples can be found in [examples/models](examples/models)

**Examples:**

*   **OpenAI-Compatible Model Evaluation:** `bash examples/models/openai_compatible.sh`
*   **vLLM Evaluation:** `bash examples/models/vllm_qwen2vl.sh`
*   **LLaVA-OneVision Evaluation:** `bash examples/models/llava_onevision.sh`
*   **Qwen2-VL Evaluation:** `bash examples/models/qwen2_vl.sh`
*   **Llama3-Vision Evaluation:** `bash examples/models/llama_vision.sh`

**More Usage:**

```bash
python3 -m lmms_eval --help
```

**Environment Variables:**

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include 
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

**Troubleshooting:**

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# If you are using numpy==2.x, sometimes may causing errors
python3 -m pip install numpy==1.26;
# Someties sentencepiece are required for tokenizer to work
python3 -m pip install sentencepiece;
```

## Customization

Refer to the [documentation](docs/README.md) for adding custom models and datasets.

## Acknowledgements

Based on the design of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

---

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