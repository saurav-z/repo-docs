<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png" alt="lmms-eval Logo">
</p>

# lmms-eval: The Premier Evaluation Suite for Large Multimodal Models

**lmms-eval** empowers the rapid development and thorough evaluation of Large Multimodal Models (LMMs) by providing a comprehensive and efficient framework. [Visit the original repository](https://github.com/EvolvingLMMs-Lab/lmms-eval).

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features:

*   **Extensive Task Support:** Evaluate LMMs across a diverse range of text, image, video, and audio tasks (100+ tasks).
*   **Wide Model Compatibility:** Supports over 30 different LMMs, including popular architectures.
*   **Efficient Evaluation Framework:** Designed for consistent and rapid assessment of LMMs.
*   **Integration with vLLM and OpenAI-compatible APIs:** Enables accelerated evaluation and supports API-based models.
*   **Reproducibility Focus:** Provides resources for reproducing LLaVA-1.5 results, with detailed environment information.
*   **Regular Updates:** Continuously updated with new tasks, models, and features.

---

## Announcements

*   **[2025-07]**: Released `lmms-eval-0.4` with major updates and improvements. See the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md) for details.
*   **[2025-07]**: Added support for [PhyX](https://phyx-bench.github.io/), a benchmark for physics-grounded reasoning.
*   **[2025-06]**: Added support for [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA), evaluating math reasoning in videos.
*   **[2025-04]**: Introduced [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/), and support for batched audio evaluations.
*   **[2025-02]**: Integrated [`vllm`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/544) for accelerated evaluation and [`openai_compatible`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/546) for OpenAI API models.

<details>
<summary>Previous Updates</summary>
- [2025-01] 🎓🎓 Released our new benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826).
- [2024-12] 🎉🎉 Presented [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296).
- [2024-11] 🔈🔊 Upgraded `lmms-eval/v0.3.0` with audio evaluation support.
- [2024-10] 🎉🎉 Welcomed [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench) for vision-centric VQA.
- [2024-10] 🎉🎉 Welcomed [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench) for fine-grained temporal understanding.
- [2024-10] 🎉🎉 Welcomed new tasks [VDC](https://rese1f.github.io/aurora-web/), [MovieChat-1K](https://rese1f.github.io/MovieChat/), and [Vinoground](https://vinoground.github.io/). Also welcomed the new models: [AuroraCap](https://github.com/rese1f/aurora) and [MovieChat](https://github.com/rese1f/MovieChat).
- [2024-09] 🎉🎉 Welcomed new tasks [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/) for inference acceleration
- [2024-09] ⚙️️⚙️️️️ Upgraded `lmms-eval` to `0.2.3` with more tasks and features, with code credit to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [2024-08] 🎉🎉 Welcomed the new model [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), new tasks [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158). We provide new feature of SGlang Runtime API for llava-onevision model, please refer the [doc](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/commands.md) for inference acceleration
- [2024-07] 👨‍💻👨‍💻 Upgraded `lmms-eval/v0.2.1` to support more models and evaluation tasks, e.g. [Details Captions](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/136), [MLVU](https://arxiv.org/abs/2406.04264), [WildVision-Bench](https://huggingface.co/datasets/WildVision/wildvision-arena-data), [VITATECS](https://github.com/lscpku/VITATECS) and [LLaVA-Interleave-Bench](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/).
- [2024-07] 🎉🎉 Released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)! 
- [2024-06] 🎬🎬 Upgraded `lmms-eval/v0.2.0` to support video evaluations.
- [2024-03] 📝📝 Released the first version of `lmms-eval`.
</details>

## Installation

### Recommended: Using `uv` (for Consistent Environments)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval

# Create and activate your environment (from uv.lock)
uv sync

# Run commands
uv run python -m lmms_eval --help

# Add dependencies
uv add <package>
```

### Alternative Installation: Direct from Git

```bash
uv venv eval
uv venv --python 3.12  # Or your preferred Python version
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

<details>
<summary>Reproducing LLaVA-1.5 Results</summary>

Refer to the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) to reproduce LLaVA-1.5 paper results.
Also, check the [results check](miscs/llava_result_check.md) for environment variations.

</details>

### Java Installation (for `pycocoeval`)

```bash
conda install openjdk=8
# check java version
java -version
```

<details>
<summary>Comprehensive Evaluation Results</summary>
Detailed results for the LLaVA series models are available in the Google Sheet [here](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing) and the raw data exported from Weights & Biases [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).
</details>

If testing [VILA](https://github.com/NVlabs/VILA), install:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

## Usage Examples

> Find more examples in the [examples/models](examples/models) directory.

**Evaluate an OpenAI-Compatible Model:**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Evaluate with vLLM:**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Evaluate LLaVA-OneVision:**

```bash
bash examples/models/llava_onevision.sh
```

**Evaluate LLaVA-OneVision1_5:**

```bash
bash examples/models/llava_onevision1_5.sh
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

**For More Parameters:**

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
# Other possibilities include ANTHROPIC_API_KEY, DASHSCOPE_API_KEY, etc.
```

## Common Issues and Solutions

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26; # if using numpy==2.x
python3 -m pip install sentencepiece; # if needed for tokenizers
```

## Add Customized Model and Dataset

See our [documentation](docs/README.md).

## Acknowledgements

`lmms-eval` is based on [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). Explore the [lm-eval-harness documentation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for related information.

---

## Key Changes from lm-eval-harness:

*   Build context passes in `idx` and processes images during model response.
*   `Instance.args` now contains a list of images.
*   Dedicated classes for each LMM due to input/output format differences.

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