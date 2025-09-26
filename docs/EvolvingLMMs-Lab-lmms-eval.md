<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png" alt="LMMs-Eval Logo">
</p>

# LMMs-Eval: Your Comprehensive Suite for Evaluating Large Multimodal Models (LMMs)

>  **Evaluate and accelerate the development of your LMMs with LMMs-Eval, a robust and versatile evaluation framework.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

[**View the Original Repository on GitHub**](https://github.com/EvolvingLMMs-Lab/lmms-eval)

LMMs-Eval is a powerful and flexible evaluation framework designed to accelerate the development of Large Multimodal Models (LMMs). It provides a unified platform for assessing LMM performance across a wide range of tasks, datasets, and models.

**Key Features:**

*   **Extensive Task Support:** Evaluate LMMs on a growing collection of **100+ tasks** spanning text, image, video, and audio modalities.
*   **Broad Model Compatibility:** Supports **30+ models**, including popular architectures and custom implementations.
*   **Efficient Evaluation:** Integrates cutting-edge technologies like `vllm` and `openai_compatible` for faster and more efficient evaluations.
*   **Reproducibility Focused:**  Includes detailed documentation, example scripts, and environment setup options to ensure reliable and reproducible results.
*   **Active Development:**  Continuously updated with new tasks, models, and features, reflecting the latest advancements in the field.

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## What's New

### Recent Updates:
-   **[2025-07]** 🚀🚀 Released `lmms-eval-0.4` with significant updates. See the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md)
-   **[2025-07]** 🎉🎉 New task [PhyX](https://phyx-bench.github.io/) added
-   **[2025-06]** 🎉🎉 New task [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA) added
-   **[2025-04]** 🚀🚀 Added support for [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/)
-   **[2025-02]** 🚀🚀 Integrated [`vllm`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/544) and [`openai_compatible`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/546)
-   **[2025-01]** 🎓🎓 New benchmark: [Video-MMMU](https://arxiv.org/abs/2501.13826) released
-   **[2024-12]** 🎉🎉 Published [MME-Survey](https://arxiv.org/pdf/2411.15296)
-   **[2024-11]** 🔈🔊 Audio evaluations supported for Qwen2-Audio and Gemini-Audio
-   **[2024-10]** 🎉🎉 Added tasks [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench), [VDC](https://rese1f.github.io/aurora-web/), [MovieChat-1K](https://rese1f.github.io/MovieChat/), [Vinoground](https://vinoground.github.io/) and Models [AuroraCap](https://github.com/rese1f/aurora) and [MovieChat](https://github.com/rese1f/MovieChat)
-   **[2024-09]** 🎉🎉 Added tasks [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/)
-   **[2024-09]** ⚙️️⚙️️️️ Upgraded `lmms-eval` to `0.2.3` and made performance enhancements
-   **[2024-08]** 🎉🎉 Added models [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), and the tasks [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158). Added SGlang Runtime API for llava-onevision model
-   **[2024-07]** 👨‍💻👨‍💻 Upgraded `lmms-eval/v0.2.1`, and supported more models and evaluation tasks.
-   **[2024-07]** 🎉🎉 Published [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)!
-   **[2024-06]** 🎬🎬  Upgraded `lmms-eval/v0.2.0` to support video evaluations.
-   **[2024-03]** 📝📝  Released the first version of `lmms-eval`

<details>
<summary>Older Updates</summary>
...
</details>

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
<summary>Reproducing LLaVA-1.5 results</summary>

...
</details>

If you need to test on caption datasets, install `java==1.8.0`:

```bash
conda install openjdk=8
```

<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>
...
</details>

If you want to test [VILA](https://github.com/NVlabs/VILA), install the following dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

## Usage Examples

**Example scripts for quickstart (more in `examples/models`)**:

*   **OpenAI-Compatible Model Evaluation:**

    ```bash
    bash examples/models/openai_compatible.sh
    bash examples/models/xai_grok.sh
    ```
*   **vLLM Evaluation:**

    ```bash
    bash examples/models/vllm_qwen2vl.sh
    ```
*   **LLaVA-OneVision Evaluation:**

    ```bash
    bash examples/models/llava_onevision.sh
    ```
*   **LLaVA-OneVision1\_5 Evaluation:**

    ```bash
    bash examples/models/llava_onevision1_5.sh
    ```
*   **LLaMA-3.2-Vision Evaluation:**

    ```bash
    bash examples/models/llama_vision.sh
    ```
*   **Qwen2-VL Evaluation:**

    ```bash
    bash examples/models/qwen2_vl.sh
    bash examples/models/qwen2_5_vl.sh
    ```
*   **LLaVA on MME:**

    ```bash
    bash examples/models/llava_next.sh
    ```
*   **Tensor Parallel Evaluation:**

    ```bash
    bash examples/models/tensor_parallel.sh
    ```
*   **SGLang Evaluation:**

    ```bash
    bash examples/models/sglang.sh
    ```
*   **vLLM Evaluation (Qwen2-VL):**

    ```bash
    bash examples/models/vllm_qwen2vl.sh
    ```

**More Parameters:**
```bash
python3 -m lmms_eval --help
```

**Environment Variables (recommended):**

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

**Common Environment Issues:**

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26;
python3 -m pip install sentencepiece;
```

## Contribute

We welcome contributions! Please refer to our [documentation](docs/README.md) for information on adding custom models and datasets.  Please share feedback, ask questions in issues/PRs on GitHub.

## Acknowledgements

This project is based on and inspired by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

---

**Citations**

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