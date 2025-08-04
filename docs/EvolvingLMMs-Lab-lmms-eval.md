<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png" alt="LMMs-Eval Logo">
</p>

# LMMs-Eval: Your Comprehensive Suite for Evaluating Large Multimodal Models (LMMs)

> **[LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) empowers researchers and developers to rigorously evaluate and advance large multimodal models across diverse tasks.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features of LMMs-Eval:

*   **Extensive Task Coverage:** Evaluate LMMs on a wide range of tasks, including text, image, video, and audio assessments.
*   **Broad Model Support:** Compatible with over 30 popular LMMs, with continuous updates for new models.
*   **Accelerated Evaluation:** Integrates technologies like `vllm` and `openai_compatible` for faster and more efficient model evaluation.
*   **Reproducibility Focus:** Offers scripts and resources to reproduce results from key LMM publications.
*   **Customization:** Easily add custom models and datasets to tailor the evaluation process to your specific needs.
*   **Regular Updates:** Stay at the forefront of LMM evaluation with frequent updates incorporating the latest benchmarks, models, and features.

---

## Announcements

*   **[2025-07]** 🚀🚀 Released `lmms-eval-0.4` with major updates and improvements; see the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md). For users of `lmms-eval-0.3`, refer to the `stable/v0d3` branch.  Join the [discussion](https://github.com/EvolvingLMMs-Lab/lmms-eval/discussions/779) for model evaluation results.
*   **[2025-04]** 🚀🚀  Added support for Aero-1-Audio, with batched evaluations.
*   **[2025-07]** 🎉🎉 Added support for the [PhyX](https://phyx-bench.github.io/) benchmark.
*   **[2025-06]** 🎉🎉 Added support for the [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA) benchmark.
*   **[2025-02]** 🚀🚀 Integrated `vllm` for accelerated evaluation and `openai_compatible` for API-based models.

<details>
<summary>Earlier Updates</summary>
  *   **[2025-01]** Released [Video-MMMU](https://arxiv.org/abs/2501.13826).
  *   **[2024-12]** Presented [MME-Survey](https://arxiv.org/pdf/2411.15296).
  *   **[2024-11]** Added audio evaluation support for models like Qwen2-Audio and Gemini-Audio.
  *   **[2024-10]** Added support for [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench), [VDC](https://rese1f.github.io/aurora-web/), [MovieChat-1K](https://rese1f.github.io/MovieChat/), and [Vinoground](https://vinoground.github.io/).  Supported new models: [AuroraCap](https://github.com/rese1f/aurora) and [MovieChat](https://github.com/rese1f/MovieChat).
  *   **[2024-09]** Added support for [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/).
  *   **[2024-09]** Upgraded to `0.2.3` with more tasks and features.
  *   **[2024-08]** Added support for [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158), and SGlang Runtime API for llava-onevision.
  *   **[2024-07]** Upgraded to `v0.2.1` with support for more models including [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA), [InternVL-2](https://github.com/OpenGVLab/InternVL), [VILA](https://github.com/NVlabs/VILA), and many more evaluation tasks, e.g. [Details Captions](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/136), [MLVU](https://arxiv.org/abs/2406.04264), [WildVision-Bench](https://huggingface.co/datasets/WildVision/wildvision-arena-data), [VITATECS](https://github.com/lscpku/VITATECS) and [LLaVA-Interleave-Bench](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/).
  *   **[2024-07]** Released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)!
  *   **[2024-06]** Upgraded to `v0.2.0` with video evaluation support for video models.
  *   **[2024-03]** Released the first version of `lmms-eval`.

</details>

---

## Installation

### Quick Installation (with `uv`)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

### For Development

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv venv dev
source dev/bin/activate
uv pip install -e .
```

<details>
<summary>Reproducing LLaVA-1.5 Results</summary>

Use the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) to reproduce LLaVA-1.5 paper results.  See [results check](miscs/llava_result_check.md) for variations due to torch/cuda versions.

</details>

**Java Dependency:**

If using caption datasets (e.g., `coco`, `refcoco`, `nocaps`), install Java 1.8:

```bash
conda install openjdk=8
```

Then verify with `java -version`.

<details>
<summary>Comprehensive Evaluation Results (LLaVA Series)</summary>

Find detailed results and dataset information in the [Google Sheet](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing). Raw data is also available [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

</details>

**VILA Dependency:**

To test [VILA](https://github.com/NVlabs/VILA):

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

---

## Usage Examples

> Find more examples in [examples/models](examples/models)

**Evaluate OpenAI-Compatible Models:**

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

(Requires cloning LLaVA repo from [LLaVA](https://github.com/haotian-liu/LLaVA))

```bash
bash examples/models/llava_next.sh
```

**Tensor Parallel Evaluation (larger models):**

```bash
bash examples/models/tensor_parallel.sh
```

**SGLang Evaluation (larger models):**

```bash
bash examples/models/sglang.sh
```

**vLLM Evaluation (larger models):**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**More Parameters:**

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
# Other API Keys (ANTHROPIC_API_KEY, DASHSCOPE_API_KEY, etc.)
```

## Common Issue Solutions

```bash
python3 -m pip install httpx==0.23.3
python3 -m pip install protobuf==3.20
python3 -m pip install numpy==1.26 # If using numpy==2.x
python3 -m pip install sentencepiece # May be needed for tokenizers
```

---

## Adding Custom Models and Datasets

See the [documentation](docs/README.md) for details.

---

## Acknowledgements

LMMs-Eval is based on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).  Refer to the [lm-evaluation-harness docs](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for additional information.

---

## Key Changes from Original API:

*   Build context only passes `idx` and processes images/docs during the model response phase.
*   `Instance.args` (in `lmms_eval/api/instance.py`) now contains a list of images.
*   Creates a new class for each LMM due to the lack of unified input/output formats in HF.

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