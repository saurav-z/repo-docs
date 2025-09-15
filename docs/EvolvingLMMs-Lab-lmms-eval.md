<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# LMMs-Eval: Comprehensive Evaluation Suite for Large Multimodal Models

**Easily benchmark and accelerate the development of your Large Multimodal Models with LMMs-Eval!**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

LMMs-Eval provides a robust and efficient framework for evaluating and comparing Large Multimodal Models (LMMs).  This suite supports a wide range of tasks across text, image, video, and audio, offering a centralized platform for assessing model performance and driving innovation in multimodal AI.  Explore the cutting-edge capabilities and stay up-to-date with the latest advancements in LMM evaluation!

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

**Key Features:**

*   **Comprehensive Task Coverage:** Evaluate LMMs across a diverse set of tasks, including text, image, video, and audio understanding.
*   **Wide Model Support:**  Compatibility with a growing library of over 30+ LMMs, including popular models.
*   **Accelerated Evaluation:** Integrated support for frameworks like `vLLM` and OpenAI API compatibility for faster model evaluation.
*   **Reproducibility:** Offers scripts and detailed environment setup for reproducing results, facilitating rigorous research.
*   **Extensible:**  Easily add support for new models and datasets, tailoring the evaluation process to specific research needs.
*   **Active Community:** Benefit from ongoing development, updates, and community support via GitHub issues and Discord.

---

## Recent Updates and Announcements

*   **[2025-07]** 🚀🚀 Released `lmms-eval-0.4` with major updates and improvements; see [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md). Discuss model evaluation results in [discussion](https://github.com/EvolvingLMMs-Lab/lmms-eval/discussions/779).
*   **[2025-07]** 🎉🎉  Support for the [PhyX](https://phyx-bench.github.io/) benchmark, focused on physics-grounded reasoning in visual scenarios.
*   **[2025-06]** 🎉🎉  Support for the [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA) benchmark, evaluating mathematical reasoning in educational videos.
*   **[2025-04]** 🚀🚀  Integration and evaluation support for the compact audio model [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/), including batched evaluations.
*   **[2025-02]** 🚀🚀  Integration of `vllm` and `openai_compatible`, enabling accelerated evaluation for both multimodal and language models.

*(See original README for a complete list of announcements)*

---

## Installation

### Using uv (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv sync  # This creates/updates your environment from uv.lock
uv run python -m lmms_eval --help  # Run any command with uv run
uv add <package>  # Updates both pyproject.toml and uv.lock
```

### Alternative Installation

```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

<details>
<summary>Reproducing LLaVA-1.5's Results</summary>
You can check the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) to **reproduce LLaVA-1.5's paper results**. We found torch/cuda versions difference would cause small variations in the results, we provide the [results check](miscs/llava_result_check.md) with different environments.
</details>

If you want to test on caption dataset such as `coco`, `refcoco`, and `nocaps`, you will need to have `java==1.8.0` to let pycocoeval api to work. If you don't have it, you can install by using conda
```
conda install openjdk=8
```
you can then check your java version by `java -version`

<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>
<br>

As demonstrated by the extensive table below, we aim to provide detailed information for readers to understand the datasets included in lmms-eval and some specific details about these datasets (we remain grateful for any corrections readers may have during our evaluation process).

We provide a Google Sheet for the detailed results of the LLaVA series models on different datasets. You can access the sheet [here](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing). It's a live sheet, and we are updating it with new results.

<p align="center" width="100%">
<img src="https://i.postimg.cc/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%">
</p>

We also provide the raw data exported from Weights & Biases for the detailed results of the LLaVA series models on different datasets. You can access the raw data [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

</details>
<br>

If you want to test [VILA](https://github.com/NVlabs/VILA), you should install the following dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

## Usage Examples

*(See original README for examples of running specific models)*

```bash
# Evaluation of OpenAI-Compatible Model
bash examples/models/openai_compatible.sh

# Evaluation of vLLM
bash examples/models/vllm_qwen2vl.sh

# ... (other examples)
```

**For more usage and parameters:**

```bash
python3 -m lmms_eval --help
```

**Environment Variables:**

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

**Common Environment Issues**
*(See original README for common environment issues and solutions)*

## Adding Custom Models and Datasets

Refer to the [documentation](docs/README.md) for instructions on adding your own models and datasets.

## Acknowledgements

LMMs-Eval is inspired by and built upon the [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) project.  We encourage you to consult the [lm-eval-harness documentation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for more information.

---

**(Changes from original API are listed in the original README)**

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