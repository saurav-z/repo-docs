<p align="center" width="70%">
<img src="https://i.postimg.com/KvkLzbF9/WX20241212-014400-2x.png" alt="LMMs-Eval Logo">
</p>

# LMMs-Eval: Your Comprehensive Suite for Evaluating Large Multimodal Models

**Tackle the complexities of LMM evaluation with LMMs-Eval, a robust and versatile toolkit designed to accelerate the development of cutting-edge multimodal AI.** [![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

➡️ **[Explore the LMMs-Eval Repository on GitHub](https://github.com/EvolvingLMMs-Lab/lmms-eval)**

**Key Features:**

*   ✅ **Extensive Task Support:** Evaluate LMMs across a broad spectrum of modalities, including text, image, video, and audio.
*   ✅ **Model Agnostic:** Supports a growing number of models (30+), with easy integration for new architectures.
*   ✅ **Reproducible Results:** Focused on providing consistent and reliable evaluation, with support for consistent environments.
*   ✅ **Accelerated Evaluation:** Integrates with vLLM and OpenAI-compatible models for faster and more efficient assessment.
*   ✅ **Community-Driven:** Benefit from continuous updates, new features, and contributions from an active community.
*   ✅ **Comprehensive Documentation:** Access detailed guides and examples to streamline your evaluation process.

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## What's New

-   **[2025-07]**:  🚀🚀 Released `lmms-eval-0.4` with significant updates and improvements. See [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
-   **[2025-07]**:  🎉🎉 Added support for [PhyX](https://phyx-bench.github.io/), a physics-grounded reasoning benchmark.
-   **[2025-06]**:  🎉🎉 Added support for [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA), a mathematical reasoning benchmark for videos.
-   **[2025-04]**:  🚀🚀 Introduced [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/) and its evaluation support, including batched evaluations.
-   **[2025-02]**:  🚀🚀 Integrated [`vllm`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/544) for accelerated evaluation and [`openai_compatible`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/546) for wider API model support.

<details>
<summary>Older Updates</summary>

*   [2025-01] 🎓🎓 New benchmark released: [Video-MMMU](https://arxiv.org/abs/2501.13826).
*   [2024-12] 🎉🎉 Presented [MME-Survey](https://arxiv.org/pdf/2411.15296) jointly with MME and OpenCompass.
*   [2024-11] 🔈🔊 Added audio evaluations for audio models with lmms-eval/v0.3.0.
*   [2024-10] 🎉🎉 New benchmark [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench) support.
*   [2024-10] 🎉🎉 New benchmark [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench) support.
*   [2024-10] 🎉🎉 New task and model support including VDC, MovieChat-1K, Vinoground, AuroraCap and MovieChat.
*   [2024-09] 🎉🎉 New task MMSearch and MME-RealWorld support and `lmms-eval` upgraded to `0.2.3`
*   [2024-08] 🎉🎉 New model and task support, including LLaVA-OneVision, Mantis, MVBench, LongVideoBench, and MMStar. SGlang Runtime API feature for llava-onevision model.
*   [2024-07] 👨‍💻👨‍💻 `lmms-eval/v0.2.1` with more model and task support.
*   [2024-07] 🎉🎉 Technical report and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench) released.
*   [2024-06] 🎬🎬 `lmms-eval/v0.2.0` with support for video evaluations.
*   [2024-03] 📝📝 Released the first version of `lmms-eval`.

</details>

## Installation

### Using uv (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv sync
```

Run commands with: `uv run python -m lmms_eval --help`
Add dependencies with: `uv add <package>`

### Alternative Installation

```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

<details>
<summary>Reproducing LLaVA-1.5 Results</summary>

Refer to the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt). See [results check](miscs/llava_result_check.md) for result variations.

</details>

Install Java 1.8.0 if needed for `coco`, `refcoco`, and `nocaps` datasets:
```bash
conda install openjdk=8
```

<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>

Access the detailed results:  [Google Sheet](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing) and [Raw Data](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

</details>

Install s2wrapper for testing VILA:
```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

## Usages

> See [examples/models](examples/models) for more.

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

**LLaVA Evaluation on MME:**
```bash
bash examples/models/llava_next.sh
```

**Tensor Parallel Evaluation (llava-next-72b):**
```bash
bash examples/models/tensor_parallel.sh
```

**SGLang Evaluation (llava-next-72b):**
```bash
bash examples/models/sglang.sh
```

**vLLM Evaluation (llava-next-72b):**
```bash
bash examples/models/vllm_qwen2vl.sh
```

**More Parameters:**  `python3 -m lmms_eval --help`

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

**Troubleshooting Common Issues:**
```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26;
python3 -m pip install sentencepiece;
```

## Adding Custom Models and Datasets

Refer to the [documentation](docs/README.md).

## Acknowledgements

LMMs-Eval is built upon [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

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
```