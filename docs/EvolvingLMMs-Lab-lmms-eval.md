<p align="center" width="70%">
<img src="https://i.postimg.com/KvkLzbF9/WX20241212-014400-2x.png" alt="lmms-eval Logo">
</p>

# lmms-eval: Evaluate and Advance Large Multimodal Models

**lmms-eval provides a comprehensive toolkit for evaluating and accelerating the development of Large Multimodal Models (LMMs).**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

Accelerate your LMM research and development with `lmms-eval`, supporting a wide range of tasks across text, image, video, and audio modalities.

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features

*   **Extensive Task Support:** Evaluate LMMs on a diverse set of over 100 tasks, including text, image, video, and audio.
*   **Broad Model Compatibility:** Supports more than 30 LMMs, including popular models like LLaVA, Qwen2-VL, and more.
*   **Flexible Evaluation:** Offers support for various evaluation backends like vLLM and OpenAI-compatible APIs.
*   **Accelerated Evaluation:** Integrates with vLLM for faster evaluation speeds.
*   **Reproducibility Focus:**  Includes scripts and resources to reproduce results and promote reliable research.
*   **Comprehensive Documentation:** Provides detailed documentation and examples to get you started quickly.
*   **Community Driven:** Benefit from contributions from the research community.

## What's New

*   **[2025-07]** - Released `lmms-eval-0.4` featuring major updates and improvements, along with a dedicated discussion thread.
*   **[2025-07]** - Added support for the [PhyX](https://phyx-bench.github.io/) benchmark.
*   **[2025-06]** - Added support for the [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA) benchmark.
*   **[2025-04]** - Introduced [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/) - a compact audio model with batched evaluation support.
*   **[2025-02]** - Integrated `vllm` and `openai_compatible` for accelerated evaluation.

For a complete list of recent updates and features, see the full [original README](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## Installation

### Using uv (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv sync  # This creates/updates your environment from uv.lock
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

For dataset testing requiring `java==1.8.0`:

```bash
conda install openjdk=8
```

## Usages

### Evaluate OpenAI-Compatible Model
```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

### Evaluate vLLM
```bash
bash examples/models/vllm_qwen2vl.sh
```

### Evaluate LLaVA-OneVision
```bash
bash examples/models/llava_onevision.sh
```

### Evaluate LLaMA-3.2-Vision
```bash
bash examples/models/llama_vision.sh
```

### Evaluate Qwen2-VL
```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

### Evaluate LLaVA on MME
```bash
bash examples/models/llava_next.sh
```

### Evaluate with tensor parallel (llava-next-72b)
```bash
bash examples/models/tensor_parallel.sh
```

### Evaluate with SGLang (llava-next-72b)
```bash
bash examples/models/sglang.sh
```

### Evaluate with vLLM (llava-next-72b)
```bash
bash examples/models/vllm_qwen2vl.sh
```

### More Parameters

```bash
python3 -m lmms_eval --help
```

### Environmental Variables

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
```

### Common Environment Issues

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26;
python3 -m pip install sentencepiece;
```

##  [Original Repository](https://github.com/EvolvingLMMs-Lab/lmms-eval)

## Add Customized Model and Dataset

Please refer to our [documentation](docs/README.md).

## Acknowledgement

lmms_eval is a fork of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). We recommend you to read through the [docs of lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for relevant information.

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