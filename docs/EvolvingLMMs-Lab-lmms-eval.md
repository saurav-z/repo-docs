<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png" alt="lmms-eval Logo">
</p>

# lmms-eval: The Comprehensive Evaluation Suite for Large Multimodal Models

**Accelerate your LMM development with `lmms-eval`, a powerful and versatile evaluation framework for large multimodal models, supporting diverse tasks and models.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> **[Original Repository](https://github.com/EvolvingLMMs-Lab/lmms-eval)**

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features

*   **Extensive Task Support:** Evaluate LMMs across a wide range of text, image, video, and audio tasks.
*   **Broad Model Compatibility:**  Supports a diverse selection of over 30 LMMs, including models from OpenAI-compatible APIs and vLLM.
*   **Reproducibility Focus:** Dedicated resources and scripts to reproduce results from leading LMMs.
*   **Accelerated Evaluation:** Integrations with vLLM and OpenAI-compatible APIs for efficient model evaluation.
*   **Active Development:**  Continuously updated with new tasks, models, and features, driven by community contributions.
*   **Comprehensive Resources:** Includes detailed documentation, examples, and a growing collection of evaluation results.

---

## Recent Updates and Announcements

*   **[2025-07]** 🚀🚀 Released `lmms-eval-0.4` with major updates; see [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md)
*   **[2025-07]** 🎉🎉 Support for [PhyX](https://phyx-bench.github.io/), a physics-grounded reasoning benchmark.
*   **[2025-06]** 🎉🎉 Support for [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA), evaluating mathematical reasoning from videos.
*   **[2025-04]** 🚀🚀  Introduced [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/) with batched evaluations support.
*   **[2025-02]** 🚀🚀  Integrated `vllm` and `openai_compatible` for accelerated evaluation.

<details>
<summary>Older Announcements</summary>

*   [2025-01] 🎓🎓 Released [Video-MMMU](https://arxiv.org/abs/2501.13826) benchmark.
*   [2024-12] 🎉🎉 Presented [MME-Survey](https://arxiv.org/pdf/2411.15296) survey on multimodal LLM evaluation.
*   [2024-11] 🔈🔊 Added audio evaluations in `lmms-eval/v0.3.0`.
*   [2024-10] 🎉🎉 Added [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench) and [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench).
*   [2024-10] 🎉🎉 Added [VDC](https://rese1f.github.io/aurora-web/), [MovieChat-1K](https://rese1f.github.io/MovieChat/), [Vinoground](https://vinoground.github.io/) tasks; [AuroraCap](https://github.com/rese1f/aurora) and [MovieChat](https://rese1f/MovieChat) models.
*   [2024-09] 🎉🎉 Added [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/) for inference acceleration.
*   [2024-09] ⚙️️️️ Upgraded `lmms-eval` to `0.2.3` with more features.
*   [2024-08] 🎉🎉  Added [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158).
*   [2024-07] 👨‍💻👨‍💻 Upgraded `lmms-eval/v0.2.1` with support for more models and tasks.
*   [2024-07] 🎉🎉 Released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench).
*   [2024-06] 🎬🎬 Upgraded `lmms-eval/v0.2.0` for video evaluations.
*   [2024-03] 📝📝 Released the first version of `lmms-eval`.

</details>

---

## Why Use `lmms-eval`?

`lmms-eval` provides a centralized and efficient platform for evaluating the capabilities of large multimodal models, offering a comprehensive set of benchmarks, tools, and resources. It aims to accelerate the progress towards Artificial General Intelligence (AGI) by:

*   **Simplifying Evaluation:** Streamlines the process of benchmarking LMMs, saving time and effort.
*   **Enabling Comparison:** Allows for easy comparison of different models and their performance on various tasks.
*   **Promoting Reproducibility:** Provides clear instructions and scripts for replicating results.
*   **Fostering Community:** Encourages collaboration and contribution to the field of LMM research.

---

## Installation

### Recommended: Using `uv` (For Consistent Environments)

`uv` ensures consistent package versions across development environments.

1.  **Install `uv`:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Clone the Repository and Sync:**
    ```bash
    git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
    cd lmms-eval
    uv sync  # Creates/updates the environment from uv.lock
    ```
3.  **Run Commands:**
    ```bash
    uv run python -m lmms_eval --help  # Run commands with uv run
    ```
4.  **Add Dependencies:**
    ```bash
    uv add <package>  # Updates pyproject.toml and uv.lock
    ```

### Alternative Installation

1.  **Create and Activate a Virtual Environment:**
    ```bash
    uv venv eval
    uv venv --python 3.12
    source eval/bin/activate
    ```
2.  **Install the Package:**
    ```bash
    uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    ```

<details>
<summary>Reproducing LLaVA-1.5 Results</summary>

Reproduce LLaVA-1.5 paper results using the [environment install script](miscs/repr_scripts.sh) and the [torch environment info](miscs/repr_torch_envs.txt). Check the [results check](miscs/llava_result_check.md) for results variation based on torch/cuda versions.

</details>

If you need to test caption datasets, you might need `java==1.8.0`. If not installed:

```bash
conda install openjdk=8
```

Check Java version with `java -version`.

<details>
<summary>Comprehensive Evaluation Results</summary>

Detailed results for LLaVA series models are available in the [Google Sheet](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing). This is a live sheet, and we update with new findings.

<img src="https://i.postimg.cc/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%" alt="LLaVA results table">

Raw data exported from Weights & Biases can be found [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

</details>

If testing [VILA](https://github.com/NVlabs/VILA):

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

---

## Usages

> Find more usage examples in [examples/models](examples/models)

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

**Evaluate LLaVA on MME (Clone LLaVA repo):**

```bash
bash examples/models/llava_next.sh
```

**Evaluate with Tensor Parallel (llava-next-72b):**

```bash
bash examples/models/tensor_parallel.sh
```

**Evaluate with SGLang (llava-next-72b):**

```bash
bash examples/models/sglang.sh
```

**Evaluate with vLLM (llava-next-72b):**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**See Available Parameters:**

```bash
python3 -m lmms_eval --help
```

---

## Environment Variables

Set these environment variables before running evaluations (some are required):

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other variables: ANTHROPIC_API_KEY, DASHSCOPE_API_KEY, etc.
```

---

## Common Environment Issues

Address common issues like `httpx` or `protobuf` errors by running:

```bash
python3 -m pip install httpx==0.23.3
python3 -m pip install protobuf==3.20
python3 -m pip install numpy==1.26  # If using numpy==2.x, could cause errors
python3 -m pip install sentencepiece  # Sometimes required for tokenizers
```

---

## Adding Custom Models and Datasets

Refer to our [documentation](docs/README.md).

---

## Acknowledgements

`lmms-eval` is based on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).  Consult the [lm-evaluation-harness docs](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for more information.

---

## API Changes

*   Context building now passes only the index and processes images and documents during model response phase. This is because datasets now contain many images that cannot be stored in documents due to memory limits.
*   `Instance.args` (`lmms_eval/api/instance.py`) now contains a list of images for LMM input.
*   Currently, separate model classes are needed for each LMM because of differing input/output formats within the Hugging Face framework. We plan to unify them in the future.

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