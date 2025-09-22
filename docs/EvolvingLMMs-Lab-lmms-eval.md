<p align="center" width="70%">
<img src="https://i.postimg.com/KvkLzbF9/WX20241212-014400-2x.png" alt="lmms-eval Logo">
</p>

# lmms-eval: The Premier Evaluation Suite for Large Multimodal Models (LMMs)

**Tackle the complexities of evaluating LMMs with `lmms-eval`, your comprehensive solution for benchmarking and advancing multimodal AI.**  Find the original repository [here](https://github.com/EvolvingLMMs-Lab/lmms-eval).

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features:

*   **Extensive Task Support:** Evaluate LMMs across a wide range of tasks, including text, image, video, and audio modalities (100+ supported tasks).
*   **Broad Model Compatibility:**  Seamlessly evaluate over 30+ leading LMMs.
*   **Accelerated Evaluation:** Integrate with `vllm` and `openai_compatible` for faster and more efficient evaluations.
*   **Reproducibility:**  Utilize `uv` for consistent environments and detailed instructions for reproducing paper results.
*   **Regular Updates:** Benefit from continuous improvements, new tasks, and model integrations.

---

## What's New: Recent Highlights

*   **[2025-07]** Released `lmms-eval-0.4` with major updates. Check the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-07]** Added support for [PhyX](https://phyx-bench.github.io/) benchmark.
*   **[2025-06]** Added support for [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA) benchmark.
*   **[2025-04]** Introduced [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/) for audio evaluation, with batched evaluations support.
*   **[2025-02]** Integrated `vllm` and `openai_compatible` for accelerated evaluation.

<details>
<summary>See more recent updates</summary>

*   **[2025-01]** Released new benchmark: [Video-MMMU](https://arxiv.org/abs/2501.13826)
*   **[2024-12]** Presented [MME-Survey](https://arxiv.org/pdf/2411.15296).
*   **[2024-11]** Support for audio evaluations with `lmms-eval/v0.3.0`
*   **[2024-10]** Supported tasks: [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench), [VDC](https://rese1f.github.io/aurora-web/), [MovieChat-1K](https://rese1f.github.io/MovieChat/), and [Vinoground](https://vinoground.github.io/).
*   **[2024-09]** Supported tasks: [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/).
*   **[2024-09]** Upgraded to `0.2.3` with more features and language tasks evaluations.
*   **[2024-08]** Supported model: [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), tasks: [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158).
*   **[2024-07]** Supported models: [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA), [InternVL-2](https://github.com/OpenGVLab/InternVL), [VILA](https://github.com/NVlabs/VILA), and evaluation tasks.
*   **[2024-07]** Released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)!
*   **[2024-06]** Supported video evaluations for video models with `lmms-eval/v0.2.0`.
*   **[2024-03]** Released the first version of `lmms-eval`.

</details>

---

## Installation

### Using uv (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv sync
uv run python -m lmms_eval --help
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

Check the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt). [Results check](miscs/llava_result_check.md) provides variations based on the environment.

</details>

If you are using caption datasets, install Java 1.8.0 using:

```bash
conda install openjdk=8
```

## Comprehensive Evaluation Results

<details>
<summary>LMMs-Eval Results</summary>
<br>

Access detailed results for the LLaVA series models. Access the sheet [here](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing).
  
<p align="center" width="100%">
<img src="https://i.postimg.cc/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%">
</p>

Raw data from Weights & Biases is available [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

</details>

If you are testing [VILA](https://github.com/NVlabs/VILA), install the following dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

---

## Usage Examples

> See more examples in [examples/models](examples/models)

**OpenAI-Compatible Model Evaluation:**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**vLLM Evaluation:**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**LLaVA-OneVision Evaluation:**

```bash
bash examples/models/llava_onevision.sh
```

**Llama-3-Vision Evaluation:**

```bash
bash examples/models/llama_vision.sh
```

**Qwen2-VL Evaluation:**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**LLaVA on MME Evaluation:**

```bash
bash examples/models/llava_next.sh
```

**Tensor Parallel Evaluation:**

```bash
bash examples/models/tensor_parallel.sh
```

**SGLang Evaluation:**

```bash
bash examples/models/sglang.sh
```

**vLLM for Tensor Parallel Evaluation:**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Additional Parameters:**

```bash
python3 -m lmms_eval --help
```

## Environment Variables

Set these environment variables before running evaluations:

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

## Troubleshooting

Common issues and solutions:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26;
python3 -m pip install sentencepiece;
```

---

## Customization

Refer to the [documentation](docs/README.md) for customizing models and datasets.

---

## Acknowledgements

`lmms-eval` is built upon the foundation of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

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