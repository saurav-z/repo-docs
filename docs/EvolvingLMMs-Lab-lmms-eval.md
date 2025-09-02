<p align="center" width="70%">
<img src="https://i.postimg.com/KvkLzbF9/WX20241212-014400-2x.png" alt="LMMs-Eval Logo">
</p>

# LMMs-Eval: The Comprehensive Evaluation Suite for Large Multimodal Models

**Evaluate and accelerate the development of your LMMs with LMMs-Eval, your go-to framework for rigorous multimodal model assessment.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

**[Visit the LMMs-Eval Repository on GitHub](https://github.com/EvolvingLMMs-Lab/lmms-eval)**

LMMs-Eval is a powerful, open-source evaluation framework designed for large multimodal models (LMMs).  Built upon the robust foundation of  `lm-evaluation-harness`, it provides a unified platform for assessing LMM performance across a wide range of tasks and modalities.  From text and images to video and audio, LMMs-Eval helps researchers and developers objectively measure and compare LMM capabilities, accelerating progress in the field.

**Key Features:**

*   **Extensive Task Support:** Evaluate your LMMs across 100+ tasks.  ([Supported Tasks](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md))
*   **Model Compatibility:** Supports 30+ models, including popular architectures.  ([Supported Models](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models))
*   **Modular Design:**  Easily add new tasks and models to suit your specific needs.
*   **Reproducibility Focus:**  Includes detailed instructions and scripts for reproducing key research results.
*   **Accelerated Evaluation:** Integration with vLLM and support for OpenAI-compatible models.
*   **Active Development:**  Regular updates with new features, benchmarks, and model support.
*   **Community Driven:**  Benefit from contributions and discussions within the active  [Discord](https://discord.gg/zdkwKUqrPy) community.
*   **Comprehensive Documentation:**  Well-documented for ease of use and customization. ([Documentation](docs/README.md))

---

## Recent Updates and Announcements

*   **[2025-07]** 🚀🚀 Released `lmms-eval-0.4` with significant updates and improvements. See the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-07]** 🎉🎉 Support for the new benchmark [PhyX](https://phyx-bench.github.io/), for physics-grounded reasoning.
*   **[2025-06]** 🎉🎉 Support for the new benchmark [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA).
*   **[2025-04]** 🚀🚀  Introduced [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/), with batched evaluations.
*   **[2025-02]** 🚀🚀 Integrated [`vllm`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/544) and [`openai_compatible`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/546).

*(See the original README for a complete changelog.)*

---

## Installation

### Recommended: Using `uv` for Environment Management

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository and set up the environment:
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv sync
```

To run commands:

```bash
uv run python -m lmms_eval --help  # Run any command with uv run
```

To add new dependencies:

```bash
uv add <package>  # Updates both pyproject.toml and uv.lock
```

### Alternative Installation

```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

*(See original README for detailed environment setup, Java dependencies, and LLaVA-1.5 reproduction instructions.)*

---

##  Usages

*(See original README for example commands)*

**Environmental Variables**
Before running experiments and evaluations, we recommend you to export following environment variables to your environment. Some are necessary for certain tasks to run.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include 
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

---

##  Citations

```
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