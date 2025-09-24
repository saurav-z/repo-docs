<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# LMMs-Eval: Comprehensive Evaluation Suite for Large Multimodal Models

**Quickly and consistently evaluate your Large Multimodal Models (LMMs) across diverse tasks with LMMs-Eval!** ([See the original repo](https://github.com/EvolvingLMMs-Lab/lmms-eval))

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

LMMs-Eval provides a robust and efficient framework for assessing the capabilities of LMMs, supporting a wide range of modalities and tasks.

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features

*   **Extensive Task Support:** Evaluate LMMs on over 100 tasks, including text, image, video, and audio benchmarks.
*   **Broad Model Compatibility:** Supports evaluation of 30+ LMMs, with easy integration for new models.
*   **Accelerated Evaluation:** Integrates with `vllm` and supports OpenAI API format for faster inference.
*   **Reproducibility Focus:** Offers scripts and resources to reproduce results, facilitating reliable research.
*   **Regular Updates:** Continuously incorporates new tasks, models, and features based on community contributions.

---

## Recent Updates and Announcements

*   **[2025-07]** 🚀🚀 Released `lmms-eval-0.4` with significant new features and improvements; see the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-07]** 🎉🎉 Added support for [PhyX](https://phyx-bench.github.io/), a physics-grounded reasoning benchmark.
*   **[2025-06]** 🎉🎉 Integrated [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA), evaluating mathematical reasoning in videos.
*   **[2025-04]** 🚀🚀 Introduced [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/) — a compact audio model, with evaluation support and batched evaluations.
*   **[2025-02]** 🚀🚀 Integrated `vllm` and `openai_compatible` for faster evaluation of multimodal and language models; see usage [here](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/miscs/model_dryruns).

<details>
<summary>Show More Announcements</summary>

*   **[2025-01]** 🎓🎓 Released [Video-MMMU](https://videommmu.github.io/), a benchmark for evaluating knowledge acquisition from multi-discipline professional videos.
*   **[2024-12]** 🎉🎉 Presented [MME-Survey](https://arxiv.org/pdf/2411.15296), a comprehensive survey on multimodal LLM evaluation.
*   **[2024-11]** 🔈🔊 Added audio evaluation support for models like Qwen2-Audio and Gemini-Audio.
*   **[2024-10]** 🎉🎉 Added support for [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench) and [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench).
*   **[2024-10]** 🎉🎉 Added support for VDC, MovieChat-1K, and Vinoground tasks and AuroraCap and MovieChat models.
*   **[2024-09]** 🎉🎉 Added support for MMSearch and MME-RealWorld tasks for inference acceleration.
*   **[2024-09]** ⚙️️⚙️️️️ Upgraded `lmms-eval` to `0.2.3` with more tasks and features, check the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/releases/tag/v0.2.3).
*   **[2024-08]** 🎉🎉 Added support for [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), Mantis and new tasks [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158).
*   **[2024-07]** 👨‍💻👨‍💻 Upgraded `lmms-eval` to `0.2.1` to support more models and evaluation tasks.
*   **[2024-07]** 🎉🎉 Released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench).
*   **[2024-06]** 🎬🎬 Upgraded `lmms-eval` to `v0.2.0` to support video evaluations.
*   **[2024-03]** 📝📝 Released the first version of `lmms-eval`.

</details>

---

## Installation

### Using `uv` (Recommended)

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, install the project:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv sync  # Install dependencies
```

Run commands using:

```bash
uv run python -m lmms_eval --help
```

To add dependencies:

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

Follow the instructions in [miscs/repr_scripts.sh] and check [miscs/repr_torch_envs.txt] to reproduce the LLaVA-1.5 results.  See [miscs/llava_result_check.md] for environment-specific result variations.

</details>

If using caption datasets (`coco`, `refcoco`, `nocaps`), install Java 1.8:

```bash
conda install openjdk=8
```

Verify installation:

```bash
java -version
```

<details>
<summary>LMMs-Eval Result Tables</summary>

Detailed LLaVA series model results are available in the Google Sheet [here](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing).
You can also find raw data exported from Weights & Biases [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

<p align="center" width="100%">
<img src="https://i.postimg.cc/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%">
</p>

</details>

To test [VILA](https://github.com/NVlabs/VILA):

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

We welcome feedback and contributions!  Please submit feature requests, bug reports, and questions via GitHub issues or pull requests.

## Usage Examples

> Find more examples in [examples/models](examples/models)

**OpenAI-Compatible Model Evaluation:**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**vLLM Evaluation:**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**LLaVA-OneVision Evaluation:**

```bash
bash examples/models/llava_onevision.sh
```

**LLaVA-OneVision1_5 Evaluation:**

```bash
bash examples/models/llava_onevision1_5.sh
```

**LLaMA-3.2-Vision Evaluation:**

```bash
bash examples/models/llama_vision.sh
```

**Qwen2-VL Evaluation:**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**LLaVA on MME:**

```bash
bash examples/models/llava_next.sh
```

**Tensor Parallel Evaluation:**

```bash
bash examples/models/tensor_parallel.sh
```

**SGLang Evaluation:**

```bash
bash examples/models/sglang.sh
```

**vLLM for bigger models:**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**For more parameters:**

```bash
python3 -m lmms_eval --help
```

**Environment Variables:**

Set these environment variables for proper functionality:

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

**Troubleshooting Common Issues:**

Resolve common issues with `httpx` and `protobuf`:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# If you are using numpy==2.x, sometimes may causing errors
python3 -m pip install numpy==1.26;
# Someties sentencepiece are required for tokenizer to work
python3 -m pip install sentencepiece;
```

---

## Add Custom Models and Datasets

See [documentation](docs/README.md).

## Acknowledgements

This project is based on the [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).  Consult the [lm-eval-harness docs](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for further information.

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