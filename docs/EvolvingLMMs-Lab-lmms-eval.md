<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# LMMs-Eval: Your Comprehensive Toolkit for Evaluating Large Multimodal Models (LMMs)

> **Effortlessly benchmark and analyze your LMMs with LMMs-Eval, the leading evaluation suite for text, image, video, and audio tasks.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

**LMMs-Eval** is a powerful and versatile evaluation framework designed to accelerate the development and understanding of Large Multimodal Models (LMMs). This toolkit provides a standardized and efficient way to benchmark your LMMs across a wide range of tasks, including text, image, video, and audio modalities.

**Key Features:**

*   ✅ **Extensive Task Support:** Evaluate your models on a diverse collection of over 100 tasks. [Supported Tasks](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md)
*   🤖 **Model Compatibility:** Seamlessly test and compare performance across 30+ supported models. [Supported Models](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models)
*   🚀 **Rapid Evaluation:** Built for efficiency, allowing for fast and repeatable evaluations.
*   📚 **Comprehensive Documentation:** Detailed guides and examples to get you started quickly.  [Documentation](docs/README.md)
*   📢 **Active Community:** Join our Discord for support, discussions, and the latest updates. [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)
*   📦 **Easy Installation:** Simple installation using `uv` for consistent environments.
*   🧪 **Reproducibility:** Includes scripts and information to reproduce results, ensuring reliability.

**Recent Updates & Announcements:**

*   **[2025-07]**: Released `lmms-eval-0.4` with major updates and improvements. See [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-04]**: Added support for Aero-1-Audio with batched evaluations.
*   **[2025-07]**: Integrated new tasks [PhyX](https://phyx-bench.github.io/) and [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA).
*   **[2025-02]**: Integrated `vllm` and `openai_compatible` for accelerated evaluation and OpenAI API compatibility.

**[View the original repository on GitHub](https://github.com/EvolvingLMMs-Lab/lmms-eval)**

---

## Installation

### Using `uv` (Recommended)

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For development with consistent environment:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv sync
```

Run commands:

```bash
uv run python -m lmms_eval --help
```

Add dependencies:

```bash
uv add <package>
```

### Alternative Installation

For direct usage from Git:

```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

<details>
<summary>Reproducing LLaVA-1.5 Results</summary>
Check the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) for reproducing LLaVA-1.5 results.
</details>

For caption datasets (`coco`, `refcoco`, `nocaps`), install `java==1.8.0` using conda if needed:

```bash
conda install openjdk=8
```

Check your java version with `java -version`.

<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>
<br>

Explore detailed results for the LLaVA series on various datasets.

*   Google Sheet: [Link](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing)
*   Raw Data (Weights & Biases): [Link](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing)
</details>
<br>

If you want to test [VILA](https://github.com/NVlabs/VILA), you should install the following dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

---

## Usages

> Find more examples in [examples/models](examples/models).

**Example Usages:**

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

*   **Llama3-Vision Evaluation:**

    ```bash
    bash examples/models/llama_vision.sh
    ```

*   **Qwen2-VL Evaluation:**

    ```bash
    bash examples/models/qwen2_vl.sh
    bash examples/models/qwen2_5_vl.sh
    ```

*   **LLaVA Evaluation on MME:**

    ```bash
    bash examples/models/llava_next.sh
    ```

*   **Tensor Parallel for Larger Models:**

    ```bash
    bash examples/models/tensor_parallel.sh
    ```

*   **SGLang for Larger Models:**

    ```bash
    bash examples/models/sglang.sh
    ```

*   **vLLM for Larger Models:**

    ```bash
    bash examples/models/vllm_qwen2vl.sh
    ```

*   **Additional Parameters:**

    ```bash
    python3 -m lmms_eval --help
    ```

---

**Environment Variables:**

Set these variables before running evaluations:

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other variables: ANTHROPIC_API_KEY, DASHSCOPE_API_KEY, etc.
```

---

**Common Environment Issues:**

Resolve potential issues related to `httpx`, `protobuf`, or `numpy`:

```bash
python3 -m pip install httpx==0.23.3
python3 -m pip install protobuf==3.20
python3 -m pip install numpy==1.26
python3 -m pip install sentencepiece
```

---

## Contributing

We welcome contributions!  Give us feedback on desired features, improvements, and ask questions via issues or pull requests on GitHub.

[Documentation](docs/README.md) has further guidance on custom models and datasets.

---

## Acknowledgements

Inspired by and forked from [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

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