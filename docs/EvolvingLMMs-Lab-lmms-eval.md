<p align="center" width="70%">
<img src="https://i.postimg.com/KvkLzbF9/WX20241212-014400-2x.png" alt="LMMs-Eval Logo">
</p>

# LMMs-Eval: Evaluate and Accelerate Large Multimodal Model Development

> **LMMs-Eval** is the comprehensive evaluation suite for Large Multimodal Models, supporting a wide array of tasks and models to accelerate your LMM research. [Explore the LMMs-Eval repository](https://github.com/EvolvingLMMs-Lab/lmms-eval).

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features of LMMs-Eval:

*   **Extensive Task Support:** Evaluate LMMs across a diverse range of text, image, video, and audio tasks.
*   **Broad Model Compatibility:** Compatible with a wide variety of LMMs, including popular models.
*   **Accelerated Evaluation:** Integrated with vLLM and OpenAI API compatibility for faster evaluation.
*   **Comprehensive Benchmarks:** Access to a comprehensive suite of benchmarks, including new and specialized tasks.
*   **Flexible and Customizable:** Easily add and integrate your own models and datasets.
*   **Reproducibility:** Includes scripts and resources to reproduce results.
*   **Active Development:** Continuously updated with new features, models, and tasks.

---

## Announcements:

*   **[2025-07]** 🚀🚀  **lmms-eval-0.4 Released!**  Major update with new features and improvements; see the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-04]** 🚀🚀  **Aero-1-Audio Support:**  Evaluation support for Aero-1-Audio and batched evaluations.
*   **[2025-07]** 🎉🎉  **New Tasks Added:**  Support for [PhyX](https://phyx-bench.github.io/) and [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA).
*   **[2025-02]** 🚀🚀  **vLLM and OpenAI Compatibility:**  Integration with vLLM for accelerated evaluation and OpenAI API compatibility for broader model support.

<details>
<summary>Older announcements</summary>

-   [2025-01] 🎓🎓  Video-MMMU Benchmark Released: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826).
-   [2024-12] 🎉🎉  MME-Survey Published: [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296).
-   [2024-11] 🔈🔊 Audio Evaluations: Support for audio evaluations with Qwen2-Audio and Gemini-Audio.
-   [2024-10] 🎉🎉 New Tasks: NaturalBench, TemporalBench, VDC, MovieChat-1K, and Vinoground.
-   [2024-09] 🎉🎉 New Tasks: MMSearch and MME-RealWorld, and Upgrade to lmms-eval 0.2.3.
-   [2024-08] 🎉🎉 New Models and Tasks: LLaVA-OneVision, Mantis, MVBench, LongVideoBench, MMStar. SGlang Runtime API for llava-onevision model.
-   [2024-07] 👨‍💻👨‍💻  lmms-eval/v0.2.1 upgrade: Support for LongVA, InternVL-2, VILA and many more evaluation tasks.
-   [2024-07] 🎉🎉  Technical Report & LiveBench Released: [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench).
-   [2024-06] 🎬🎬  lmms-eval/v0.2.0 upgrade:  Video evaluations for video models like LLaVA-NeXT Video and Gemini 1.5 Pro across tasks.
-   [2024-03] 📝📝  First version of `lmms-eval` released!

</details>

## Installation

**Prerequisites:** Python 3.12

**Using `uv`:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

**For Development:**

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv venv dev
source dev/bin/activate
uv pip install -e .
```

<details>
<summary>Reproducing LLaVA-1.5 Paper Results</summary>

Follow the instructions in [miscs/repr_scripts.sh](miscs/repr_scripts.sh) and use the information in [miscs/repr_torch_envs.txt](miscs/repr_torch_envs.txt) to reproduce LLaVA-1.5's paper results. Check [miscs/llava_result_check.md](miscs/llava_result_check.md) for results with different environments.

</details>

If you need to test on caption datasets like `coco`, `refcoco`, and `nocaps`, you'll need `java==1.8.0`:

```bash
conda install openjdk=8
```

Verify with `java -version`.

<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>
<br>

Detailed results and dataset information are available in this [Google Sheet](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing) which we are regularly updating.

<p align="center" width="100%">
<img src="https://i.postimg.cc/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%">
</p>

You can also view the raw data from Weights & Biases [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

</details>
<br>

**For VILA:**

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

We welcome your feedback and contributions.

## Usage Examples

> More examples can be found in [examples/models](examples/models)

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

**LLaMA-3.2-Vision Evaluation:**

```bash
bash examples/models/llama_vision.sh
```

**Qwen2-VL Evaluation:**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**LLaVA on MME Evaluation:**

If you want to test LLaVA 1.5, you will have to clone their repo from [LLaVA](https://github.com/haotian-liu/LLaVA) and

```bash
bash examples/models/llava_next.sh
```

**Tensor Parallel Evaluation (for larger models, e.g., llava-next-72b):**

```bash
bash examples/models/tensor_parallel.sh
```

**SGLang Evaluation (for larger models, e.g., llava-next-72b):**

```bash
bash examples/models/sglang.sh
```

**vLLM Evaluation (for larger models, e.g., llava-next-72b):**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**More Parameters**

```bash
python3 -m lmms_eval --help
```

## Environment Variables

Set these environment variables before running experiments:

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

## Common Environment Issues:

Resolve common errors like httpx or protobuf issues:

```bash
python3 -m pip install httpx==0.23.3
python3 -m pip install protobuf==3.20
# If you are using numpy==2.x, sometimes may causing errors
python3 -m pip install numpy==1.26
# Someties sentencepiece are required for tokenizer to work
python3 -m pip install sentencepiece
```

## Contribute

See [documentation](docs/README.md) for guidance on adding your models and datasets.

## Acknowledgements

LMMs-Eval is based on [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). Consult the [lm-eval-harness docs](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for helpful information.

---

**Key Changes to the Original API:**

*   Image and document processing happens during model response to reduce memory usage.
*   `Instance.args` now contains image lists for LMM input.
*   New classes for each LMM model due to input/output format differences; this will be addressed in the future.

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