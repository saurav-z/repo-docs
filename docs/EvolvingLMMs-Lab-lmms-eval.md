<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# lmms-eval: Evaluate and Accelerate Large Multimodal Models

**Easily benchmark and improve your Large Multimodal Models with `lmms-eval`, a comprehensive evaluation suite.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

**[View the original repository on GitHub](https://github.com/EvolvingLMMs-Lab/lmms-eval)**

---

## Key Features of lmms-eval:

*   **Extensive Task Support:** Evaluate LMMs across text, image, video, and audio tasks (100+ supported).
*   **Wide Model Compatibility:** Supports over 30 popular LMMs, with continuous updates.
*   **Accelerated Evaluation:** Integrated with `vllm` and `openai_compatible` for efficient evaluation.
*   **Reproducibility Focus:**  Provides environment setup scripts and result checking for reliable results.
*   **Customization:** Easily add custom models and datasets to fit your needs.
*   **Active Community:**  Benefit from ongoing development and contributions from a dedicated community.

## What is lmms-eval?

`lmms-eval` is an evaluation framework designed to streamline the benchmarking of Large Multimodal Models (LMMs). Building on the successful design of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), `lmms-eval` provides a centralized platform for evaluating the performance of LMMs on a wide range of tasks. It simplifies the process of assessing and comparing different models, accelerating research and development in the field of multimodal AI. This project is being actively developed by the [EvolvingLMMs-Lab](https://www.lmms-lab.com/).

## Recent Updates & Announcements:

*   **[2025-07]** 🚀🚀  `lmms-eval-0.4` Released!  Major update with new features and improvements. See the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-07]** 🎉🎉  Support for the [PhyX](https://phyx-bench.github.io/) benchmark for physics-grounded reasoning.
*   **[2025-06]** 🎉🎉 Support for the [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA) benchmark for mathematical reasoning in videos.
*   **[2025-04]** 🚀🚀  Evaluation support for [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/), featuring batched evaluations.
*   **[2025-02]** 🚀🚀  Integration of `vllm` for accelerated evaluation and `openai_compatible` for API-based models.

<details>
<summary>See more updates</summary>

*   [2025-01] 🎓🎓 [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826).
*   [2024-12] 🎉🎉 [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296).
*   [2024-11] 🔈🔊 Audio Evaluations support.
*   [2024-10] 🎉🎉 New tasks: [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench).
*   [2024-10] 🎉🎉 New tasks: [VDC](https://rese1f.github.io/aurora-web/), [MovieChat-1K](https://rese1f.github.io/MovieChat/), and [Vinoground](https://vinoground.github.io/).
*   [2024-09] 🎉🎉 New tasks: [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/).
*   [2024-09] ⚙️️️️ Upgrade to `0.2.3` with more tasks and features.
*   [2024-08] 🎉🎉 New models: [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162). new tasks [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158).
*   [2024-07] 👨‍💻👨‍💻 Upgrade to `0.2.1`.
*   [2024-07] 🎉🎉 [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)!
*   [2024-06] 🎬🎬 Support for video evaluations across tasks and models.
*   [2024-03] 📝📝 First version of `lmms-eval`.
</details>

## Installation:

### Using uv (Recommended)

Use `uv` for consistent package management:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv sync  # Install dependencies
```

Run commands with:

```bash
uv run python -m lmms_eval --help
```

Add dependencies with:

```bash
uv add <package>
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

See the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) to reproduce LLaVA-1.5's results.
</details>

```bash
conda install openjdk=8 # if needed
```

<details>
<summary>Evaluation Results</summary>

Access detailed results of the LLaVA series models on different datasets:

*   [Google Sheet](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing)
*   [Raw Data](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing)

</details>

If you want to test [VILA](https://github.com/NVlabs/VILA), you should install the following dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

## Usage Examples:

> More examples can be found in [examples/models](examples/models)

**Evaluate OpenAI-Compatible Model:**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Evaluate vLLM:**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Evaluate LLaVA-OneVision:**

```bash
bash examples/models/llava_onevision.sh
```

**Evaluate LLaMA-3.2-Vision:**

```bash
bash examples/models/llama_vision.sh
```

**Evaluate Qwen2-VL:**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**Evaluate LLaVA on MME:**

```bash
bash examples/models/llava_next.sh
```

**Evaluate with tensor parallel for bigger model (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**Evaluate with SGLang for bigger model (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**Evaluate with vLLM for bigger model (llava-next-72b)**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**More Parameters**

```bash
python3 -m lmms_eval --help
```

## Environment Variables:

Set these before running evaluations (required for some tasks):

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

## Common Issues and Solutions:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26;
python3 -m pip install sentencepiece;
```

## Adding Custom Models and Datasets:

Refer to our [documentation](docs/README.md).

## Acknowledgements

`lmms-eval` is a fork of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

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