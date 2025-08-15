<p align="center" width="70%">
<img src="https://i.postimg.com/KvkLzbF9/WX20241212-014400-2x.png" alt="lmms-eval logo">
</p>

# lmms-eval: The Comprehensive Evaluation Suite for Large Multimodal Models

**Quickly and reliably evaluate your Large Multimodal Models (LMMs) with `lmms-eval`, the go-to framework for consistent and efficient benchmarking.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

>  Accelerate LMM development with a powerful suite that supports a wide range of tasks across text, image, video, and audio modalities.

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md) |  [GitHub Repository](https://github.com/EvolvingLMMs-Lab/lmms-eval)

---

## Key Features of `lmms-eval`

*   **Extensive Task Coverage:** Evaluate LMMs on over 100 diverse tasks spanning text, image, video, and audio.
*   **Broad Model Support:**  Compatible with 30+ LMMs, including popular architectures and emerging models.
*   **Flexible Evaluation:** Supports various evaluation modes, including OpenAI-compatible APIs, vLLM, and SGLang.
*   **Reproducible Results:**  Includes detailed documentation, environment setup scripts, and result comparisons for LLaVA models.
*   **Community-Driven:** Benefit from an active community with ongoing updates and support.

## Recent Updates and Announcements

*   **[2024-07] 🚀🚀** Released `lmms-eval-0.4` with significant improvements. See the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-07] 🎉🎉** Added support for the [PhyX](https://phyx-bench.github.io/) benchmark.
*   **[2025-06] 🎉🎉** Added support for the [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA) benchmark.
*   **[2025-04] 🚀🚀** Introduced Aero-1-Audio — a compact audio model with batched evaluations.
*   **[2025-02] 🚀🚀** Integrated `vllm` and `openai_compatible` for accelerated and flexible model evaluation.

For more details, see the complete list of updates in the original [README](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## Installation

**Prerequisites:**  Python 3.12 is recommended.

**Installation using `uv` and `pip` (Recommended):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

**Installation for Development:**

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv venv dev
source dev/bin/activate
uv pip install -e .
```

**Additional Dependencies**

If testing on caption datasets (e.g., `coco`, `refcoco`, `nocaps`):

```bash
conda install openjdk=8
```

For testing [VILA](https://github.com/NVlabs/VILA):

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

## Reproducing LLaVA-1.5 Results

Refer to the scripts at `miscs/repr_scripts.sh` and `miscs/repr_torch_envs.txt` to reproduce LLaVA-1.5 paper results.  Check `miscs/llava_result_check.md` for environment variations.

## Detailed Results

Detailed evaluation results for LLaVA family models are available:

*   **Google Sheet:** [LMMs-Eval Results](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing)
*   **Weights & Biases Raw Data:** [Raw Data](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing)

## Usages

See [examples/models](examples/models) for detailed usage examples:

*   **OpenAI-Compatible Models:**  `bash examples/models/openai_compatible.sh` and `bash examples/models/xai_grok.sh`
*   **vLLM:**  `bash examples/models/vllm_qwen2vl.sh`
*   **LLaVA-OneVision:** `bash examples/models/llava_onevision.sh`
*   **LLaMA-3.2-Vision:** `bash examples/models/llama_vision.sh`
*   **Qwen2-VL:**  `bash examples/models/qwen2_vl.sh` and `bash examples/models/qwen2_5_vl.sh`
*   **LLaVA (on MME):** `bash examples/models/llava_next.sh`
*   **Tensor Parallel (llava-next-72b):** `bash examples/models/tensor_parallel.sh`
*   **SGLang (llava-next-72b):** `bash examples/models/sglang.sh`
*   **vLLM (llava-next-72b):** `bash examples/models/vllm_qwen2vl.sh`

**Command-line Help:** `python3 -m lmms_eval --help`

**Environment Variables:**
Set environment variables before running evaluations, especially for API keys and cache paths.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

**Troubleshooting:**

Common issues related to `httpx`, `protobuf`, `numpy`, and `sentencepiece`:

```bash
python3 -m pip install httpx==0.23.3
python3 -m pip install protobuf==3.20
python3 -m pip install numpy==1.26  # If using numpy==2.x, it may cause errors
python3 -m pip install sentencepiece # Sometimes required for tokenizer
```

## Contributing and Documentation

*   **Add Custom Models and Datasets:** Consult the detailed [documentation](docs/README.md).
*   **Give Feedback:**  Report issues or contribute via Pull Requests on GitHub.

## Acknowledgements

This project is based on and extends the functionality of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

---

## Citations

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