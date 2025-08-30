<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png" alt="lmms-eval Logo">
</p>

# lmms-eval: The Premier Evaluation Suite for Large Multimodal Models (LMMs)

**Accelerate LMM development with lmms-eval, a comprehensive evaluation framework supporting a vast array of text, image, video, and audio tasks.**  ([View the Repository on GitHub](https://github.com/EvolvingLMMs-Lab/lmms-eval))

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features of lmms-eval:

*   **Extensive Task Coverage:** Evaluate LMMs across a diverse range of text, image, video, and audio tasks.
*   **Broad Model Support:**  Compatible with over 30 leading LMMs, with new models consistently being added.
*   **Efficient Evaluation:** Designed for streamlined and rapid evaluation of LMMs.
*   **Reproducibility:**  Focus on ensuring reproducible results through detailed documentation and environment management.
*   **Active Development:** Benefit from continuous updates, new features, and contributions from the community.
*   **Integration with vLLM and OpenAI-Compatible APIs:** Accelerate inference and easily evaluate models via API endpoints.
*   **Comprehensive Results:** Provides detailed results, including a Google Sheet with LLaVA series model performance data.

---

## Recent Updates & Announcements

*   **[2025-07]** 🚀🚀 `lmms-eval-0.4` Release: Major update with new features and improvements. See the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-07]** 🎉🎉 New Task: [PhyX](https://phyx-bench.github.io/), a benchmark for physics-grounded reasoning.
*   **[2025-06]** 🎉🎉 New Task: [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA), a benchmark for mathematical reasoning in videos.
*   **[2025-04]** 🚀🚀 Introducing [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/) — official evaluation support.
*   **[2025-02]** 🚀🚀 Integrated `vllm` and `openai_compatible` for accelerated evaluation.

<details>
<summary>Older Announcements</summary>

*   [2025-01] 🎓🎓 Released new benchmark: [Video-MMMU](https://arxiv.org/abs/2501.13826).
*   [2024-12] 🎉🎉 Presented [MME-Survey](https://arxiv.org/pdf/2411.15296).
*   [2024-11] 🔈🔊 Upgraded `lmms-eval/v0.3.0` to support audio evaluations.
*   [2024-10] 🎉🎉 Added new tasks: [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench), [VDC](https://rese1f.github.io/aurora-web/), [MovieChat-1K](https://rese1f.github.io/MovieChat/), and [Vinoground](https://vinoground.github.io/).
*   [2024-09] 🎉🎉 New tasks: [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/)
*   [2024-09] ⚙️️⚙️️️️ Upgraded to `0.2.3` with more tasks and features.
*   [2024-08] 🎉🎉 New model: [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), new tasks [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158).
*   [2024-07] 👨‍💻👨‍💻 Upgraded `lmms-eval/v0.2.1` to support more models and evaluation tasks.
*   [2024-07] 🎉🎉 Released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)!
*   [2024-06] 🎬🎬 Upgraded `lmms-eval/v0.2.0` to support video evaluations.
*   [2024-03] 📝📝 Released the first version of `lmms-eval`.

</details>

## Installation

### Recommended: Using `uv` for Consistent Environments

Use `uv` for dependable package management and identical environments across developers.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and navigate to the repository
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval

# Sync the environment
uv sync
```

Run commands using `uv run`:

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

Check [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt). We've also included [results check](miscs/llava_result_check.md) for potential variations.

</details>

**Dependencies for Caption Datasets (e.g., `coco`, `refcoco`, `nocaps`):**

You'll need `java==1.8.0`:

```bash
conda install openjdk=8
```

Verify with `java -version`.

<details>
<summary>Comprehensive Evaluation Results (LLaVA Series)</summary>
<br>

Find detailed results for the LLaVA series models on different datasets:

*   [Google Sheet](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing)
*   [Raw Data from Weights & Biases](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing)

<p align="center" width="100%">
<img src="https://i.postimg.cc/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%">
</p>

</details>

**Dependencies for VILA:**

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

## Usage Examples

> Find more examples in [examples/models](examples/models).

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

**Llama-3.2-Vision Evaluation:**

```bash
bash examples/models/llama_vision.sh
```

**Qwen2-VL Evaluation:**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**LLaVA on MME Evaluation:**

If testing LLaVA 1.5, clone the repo from [LLaVA](https://github.com/haotian-liu/LLaVA):

```bash
bash examples/models/llava_next.sh
```

**Tensor Parallel Evaluation (for larger models):**

```bash
bash examples/models/tensor_parallel.sh
```

**SGLang Evaluation (for larger models):**

```bash
bash examples/models/sglang.sh
```

**vLLM Evaluation (for larger models):**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Explore Parameters:**

```bash
python3 -m lmms_eval --help
```

## Environment Variables

Set these variables before running evaluations:

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other: ANTHROPIC_API_KEY, DASHSCOPE_API_KEY, etc.
```

## Troubleshooting Common Issues

Resolve common errors (e.g., `httpx` or `protobuf` errors) with:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26; # If using numpy==2.x
python3 -m pip install sentencepiece; # If required by tokenizer
```

## Adding Custom Models and Datasets

Refer to the [documentation](docs/README.md).

## Acknowledgements

Based on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).  Consult the [lm-evaluation-harness documentation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for details.

---

**Key API Modifications:**

*   Build context now passes only the index and processes image/doc during the model response phase.
*   `Instance.args` (`lmms_eval/api/instance.py`) contains a list of images.
*   Due to HF model format differences, a separate class is created for each LMM model (future unification planned).

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