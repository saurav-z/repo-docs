<p align="center" width="70%">
<img src="https://i.postimg.com/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# LMMs-Eval: Your Comprehensive Toolkit for Evaluating Large Multimodal Models

**LMMs-Eval** is a powerful and versatile evaluation suite designed to accelerate the development of large multimodal models (LMMs) by offering a unified platform for rigorous testing across various tasks.  [Explore the original repository](https://github.com/EvolvingLMMs-Lab/lmms-eval).

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features

*   **Extensive Task Support:**  Evaluate your LMMs on a wide range of text, image, video, and audio tasks.
*   **Broad Model Compatibility:**  Supports evaluation for 30+ LMMs, including models from various providers.
*   **Easy Installation:**  Offers straightforward installation using `uv` for consistent environments, and alternative installation methods.
*   **Reproducibility:** Provides detailed instructions and scripts for reproducing paper results (e.g., LLaVA-1.5).
*   **Flexible Usage:** Includes example scripts for evaluating OpenAI-compatible models, vLLM, and specific LMMs (e.g., LLaVA-OneVision, Qwen2-VL).
*   **Customization:**  Documentation to add custom models and datasets.
*   **Accelerated Evaluation:** Integrated vLLM and support for OpenAI API-based models for faster evaluation.

---

## Announcements

*   **[2025-07]**: Released `lmms-eval-0.4` with major updates; see [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md). Discuss model evaluation results in the [discussion](https://github.com/EvolvingLMMs-Lab/lmms-eval/discussions/779).
*   **[2025-07]**: Added the [PhyX](https://phyx-bench.github.io/) benchmark for physics-grounded reasoning.
*   **[2025-06]**: Added the [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA) benchmark for video-based mathematical reasoning.
*   **[2025-04]**: Introduced [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/) with batched evaluations support.
*   **[2025-02]**: Integrated `vllm` for accelerated evaluation and `openai_compatible` for any OpenAI API-based models.

---

## Installation

### Using `uv` (Recommended)

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
<summary>Reproduction of LLaVA-1.5's paper results</summary>

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

> More examples can be found in [examples/models](examples/models)

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

If you want to test LLaVA 1.5, you will have to clone their repo from [LLaVA](https://github.com/haotian-liu/LLaVA) and

```bash
bash examples/models/llava_next.sh
```

**Evaluate with tensor parallel for bigger model (llava-next-72b):**

```bash
bash examples/models/tensor_parallel.sh
```

**Evaluate with SGLang for bigger model (llava-next-72b):**

```bash
bash examples/models/sglang.sh
```

**Evaluate with vLLM for bigger model (llava-next-72b):**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**More Parameters:**

```bash
python3 -m lmms_eval --help
```

## Environment Variables

Set these environment variables for optimal functionality:

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include 
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

## Common Issues and Solutions

Address potential errors with these solutions:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26;  # if using numpy 2.x, can cause issues
python3 -m pip install sentencepiece; # required for tokenizer
```

## Adding Customized Model and Dataset

Refer to our [documentation](docs/README.md) for detailed instructions.

## Acknowledgements

This project is a fork of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).  Consult the [lm-eval-harness docs](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for further information.

---

## Changes from Original API

*   Context building now only passes in `idx` and processes images/docs during model response to prevent CPU memory overflow.
*   `Instance.args` now contains a list of images for LMM input.
*   lmms requires separate classes for each model due to input/output format differences; efforts are underway to unify this.

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