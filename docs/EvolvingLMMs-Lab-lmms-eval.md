<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png" alt="LMMs-Eval Logo">
</p>

# LMMs-Eval: Your Toolkit for Evaluating Large Multimodal Models

**Accelerate your LMM development with LMMs-Eval, a comprehensive evaluation suite for text, image, video, and audio tasks.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

**[Original Repository](https://github.com/EvolvingLMMs-Lab/lmms-eval)** | 🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://discord.gg/zdkwKUqrPy"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features

*   **Extensive Task Support:** Evaluate LMMs across text, image, video, and audio domains.
*   **Broad Model Compatibility:** Supports over 30 different models, including popular architectures.
*   **Comprehensive Evaluation:** Access to over 100 tasks and datasets.
*   **Flexible Installation:** Supports installation via `uv` (recommended) and direct git usage.
*   **Easy to Use:** Provides example scripts for various model evaluations.
*   **Accelerated Evaluation:** Integrations with `vllm` and OpenAI-compatible API support.
*   **Reproducibility:** Includes environment setup scripts to reproduce paper results.
*   **Community Driven:** Actively updated with new tasks, models, and features thanks to our contributors.

---

## Recent Updates & Announcements

-   [2025-07] 🚀🚀 Released `lmms-eval-0.4` with significant updates; see the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
-   [2025-07] 🎉🎉 Support for [PhyX](https://phyx-bench.github.io/), a physics-grounded reasoning benchmark.
-   [2025-06] 🎉🎉 Support for [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA), a video-based mathematical reasoning benchmark.
-   [2025-04] 🚀🚀 Added support for [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/) audio model, including batched evaluations.
-   [2025-02] 🚀🚀 Integrated `vllm` and `openai_compatible` for accelerated and API-based model evaluations.

<details>
<summary>See older announcements...</summary>
  ... (Previous announcements are included in the original content)
</details>

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
<summary>Reproducing LLaVA-1.5 Paper Results</summary>

Follow the instructions in [miscs/repr_scripts.sh](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/miscs/repr_scripts.sh) and review [miscs/repr_torch_envs.txt](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/miscs/repr_torch_envs.txt) for setup details.  Check [miscs/llava_result_check.md](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/miscs/llava_result_check.md) to account for small result variations due to environment differences.

</details>

If you need to evaluate caption datasets like `coco`, `refcoco`, and `nocaps`, make sure you have `java==1.8.0` installed. You can install it using conda:

```bash
conda install openjdk=8
```

Then, verify the installation with `java -version`.

<details>
<summary>Comprehensive Evaluation Results</summary>

[Access detailed results for LLaVA series models on a Google Sheet](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing).  Raw data is also available [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

</details>

If you want to test [VILA](https://github.com/NVlabs/VILA), install:
```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

## Usage Examples

> More examples can be found in [examples/models](examples/models)

**Evaluation of OpenAI-Compatible Model**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Evaluation of vLLM**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Evaluation of LLaVA-OneVision**

```bash
bash examples/models/llava_onevision.sh
```

**Evaluation of LLaMA-3.2-Vision**

```bash
bash examples/models/llama_vision.sh
```

**Evaluation of Qwen2-VL**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**Evaluation of LLaVA on MME**

```bash
bash examples/models/llava_next.sh
```

**Evaluation with tensor parallel**

```bash
bash examples/models/tensor_parallel.sh
```

**Evaluation with SGLang**

```bash
bash examples/models/sglang.sh
```

**Evaluation with vLLM**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**More Parameters**

```bash
python3 -m lmms_eval --help
```

## Environment Variables

Set these environment variables before running:

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
```

## Common Issues

Solve common issues like `httpx` or `protobuf` errors by:

```bash
python3 -m pip install httpx==0.23.3
python3 -m pip install protobuf==3.20
python3 -m pip install numpy==1.26  # If you are using numpy==2.x, sometimes may causing errors
python3 -m pip install sentencepiece # Someties sentencepiece are required for tokenizer to work
```

## Adding Custom Models and Datasets

Refer to our detailed [documentation](docs/README.md).

## Acknowledgements

This project is a fork of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).  Consult the [lm-eval-harness documentation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for more information.

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