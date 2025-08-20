<p align="center" width="70%">
<img src="https://i.postimg.com/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# LMMs-Eval: The Ultimate Evaluation Suite for Large Multimodal Models

> **LMMs-Eval empowers researchers and developers to rigorously evaluate and advance Large Multimodal Models (LMMs) across diverse tasks.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

**[View the original repository on GitHub](https://github.com/EvolvingLMMs-Lab/lmms-eval)**

LMMs-Eval is a comprehensive evaluation framework designed to accelerate the development of Large Multimodal Models (LMMs). It provides a unified platform for assessing LMMs across a wide range of tasks, supporting various modalities and models.

**Key Features:**

*   **Extensive Task Support:**  Evaluates models on over 100+ tasks, covering text, image, video, and audio.
*   **Broad Model Compatibility:** Supports 30+ LMMs, including popular architectures and open-source models.
*   **Modular and Extensible:** Easily add new tasks, models, and datasets.
*   **Reproducibility Focused:** Includes scripts and instructions to reproduce results from key LMM papers.
*   **Accelerated Evaluation:** Integrates with vLLM for faster inference and supports OpenAI API-compatible models.
*   **Active Community:** Benefit from contributions and updates from a growing community of researchers and developers.
*   **Comprehensive Documentation:** Detailed documentation and examples for easy setup and usage.
*   **Live Benchmarks:** Access to LiveBench and a Google Sheet for detailed results of the LLaVA series models on different datasets.

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Announcements

*   **[2025-07]** 🚀🚀 `lmms-eval-0.4` is released with new features and improvements.  See the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-04]** 🚀🚀 Aero-1-Audio support with batched evaluations.
*   **[2025-07]** 🎉🎉 Added [PhyX](https://phyx-bench.github.io/) benchmark.
*   **[2025-06]** 🎉🎉 Added [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA) benchmark.
*   **[2025-02]** 🚀🚀 Integrated [`vllm`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/544) and [`openai_compatible`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/546).
*   **(Older announcements collapsed for brevity - see original README for full history)**

## Installation

**Install using `uv` (Recommended):**

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

**Dependencies:**

*   For caption dataset support (`coco`, `refcoco`, `nocaps`), install Java 1.8:
    ```bash
    conda install openjdk=8
    ```
*   If testing [VILA](https://github.com/NVlabs/VILA), install:
    ```bash
    pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
    ```

<details>
<summary>Reproducing LLaVA-1.5 Results</summary>

*   Use the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) to reproduce LLaVA-1.5's results.
*   Check the [results check](miscs/llava_result_check.md) for environment variations.

</details>

<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>

*   Access a Google Sheet for detailed results: [LMMs-Eval Results](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing).
*   View raw data from Weights & Biases: [Raw Data](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

</details>

## Usages

> More examples can be found in [examples/models](examples/models)

**Example evaluation commands for various models:**

*   **OpenAI-Compatible Model:**
    ```bash
    bash examples/models/openai_compatible.sh
    bash examples/models/xai_grok.sh
    ```

*   **vLLM:**
    ```bash
    bash examples/models/vllm_qwen2vl.sh
    ```

*   **LLaVA-OneVision:**
    ```bash
    bash examples/models/llava_onevision.sh
    ```

*   **LLaMA-3.2-Vision:**
    ```bash
    bash examples/models/llama_vision.sh
    ```

*   **Qwen2-VL:**
    ```bash
    bash examples/models/qwen2_vl.sh
    bash examples/models/qwen2_5_vl.sh
    ```

*   **LLaVA on MME:**
    ```bash
    bash examples/models/llava_next.sh
    ```

*   **Tensor Parallel (LLaVA-NeXT-72b):**
    ```bash
    bash examples/models/tensor_parallel.sh
    ```

*   **SGLang (LLaVA-NeXT-72b):**
    ```bash
    bash examples/models/sglang.sh
    ```

*   **vLLM (LLaVA-NeXT-72b):**
    ```bash
    bash examples/models/vllm_qwen2vl.sh
    ```

**More Parameters**

```bash
python3 -m lmms_eval --help
```

**Environment Variables**

Set these environment variables before running evaluations (some are required):

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

**Common Issues & Solutions:**

Address potential `httpx`, `protobuf`, and `numpy` errors by:

```bash
python3 -m pip install httpx==0.23.3
python3 -m pip install protobuf==3.20
python3 -m pip install numpy==1.26
python3 -m pip install sentencepiece  # If tokenizer errors
```

## Contributing

We welcome contributions!  Please provide feedback, request features, or submit pull requests on GitHub.  See the [documentation](docs/README.md) for adding custom models and datasets.

## Acknowledgements

LMMs-Eval is a fork of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

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