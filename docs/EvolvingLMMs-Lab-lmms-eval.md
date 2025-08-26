<p align="center" width="70%">
<img src="https://i.postimg.com/KvkLzbF9/WX20241212-014400-2x.png" alt="lmms-eval Logo">
</p>

# lmms-eval: The Comprehensive Evaluation Suite for Large Multimodal Models (LMMs)

**Unlock the power of multimodal AI!** lmms-eval is your go-to framework for rigorously evaluating Large Multimodal Models (LMMs), providing a standardized and efficient way to measure performance across diverse tasks.

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> **Accelerate LMM development with a robust evaluation framework.**

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features:

*   **Extensive Task Coverage:** Evaluate LMMs across a wide range of text, image, video, and audio tasks.
*   **Broad Model Support:** Compatible with over 30 different LMMs, with new models added regularly.
*   **Efficient Evaluation:** Optimized for consistent and efficient benchmarking of LMMs.
*   **Reproducibility:** Provides environment setup instructions for reproducible results.
*   **Integration with Cutting-Edge Technologies:** Supports vLLM, and OpenAI-compatible APIs for accelerated and versatile evaluation.
*   **Community Driven:** Actively maintained and updated with contributions from the LMMs community.

## What's New?

*   **lmms-eval-0.4** - Major Update ([Release Notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md))
*   Support for new tasks: [PhyX](https://phyx-bench.github.io/), [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA).
*   Support for Aero-1-Audio, batched evaluations, and integration of vLLM and OpenAI-compatible APIs.

<details>
<summary>Recent Updates and Announcements</summary>

*   [2025-01] 🎓🎓 We have released our new benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826). Please refer to the [project page](https://videommmu.github.io/) for more details.
*   [2024-12] 🎉🎉 We have presented [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296), jointly with [MME Team](https://github.com/BradyFU/Video-MME) and [OpenCompass Team](https://github.com/open-compass).
*   [2024-11] 🔈🔊 The `lmms-eval/v0.3.0` has been upgraded to support audio evaluations for audio models like Qwen2-Audio and Gemini-Audio across tasks such as AIR-Bench, Clotho-AQA, LibriSpeech, and more. Please refer to the [blog](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.3.md) for more details!
*   [2024-10] 🎉🎉 We welcome the new task [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), a vision-centric VQA benchmark (NeurIPS'24) that challenges vision-language models with simple questions about natural imagery.
*   [2024-10] 🎉🎉 We welcome the new task [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench) for fine-grained temporal understanding and reasoning for videos, which reveals a huge (>30%) human-AI gap.
*   [2024-10] 🎉🎉 We welcome the new tasks [VDC](https://rese1f.github.io/aurora-web/) for video detailed captioning, [MovieChat-1K](https://rese1f.github.io/MovieChat/) for long-form video understanding, and [Vinoground](https://vinoground.github.io/), a temporal counterfactual LMM benchmark composed of 1000 short natural video-caption pairs. We also welcome the new models: [AuroraCap](https://github.com/rese1f/aurora) and [MovieChat](https://rese1f/MovieChat).
*   [2024-09] 🎉🎉 We welcome the new tasks [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/) for inference acceleration
*   [2024-09] ⚙️️⚙️️️️ We upgrade `lmms-eval` to `0.2.3` with more tasks and features. We support a compact set of language tasks evaluations (code credit to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)), and we remove the registration logic at start (for all models and tasks) to reduce the overhead. Now `lmms-eval` only launches necessary tasks/models. Please check the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/releases/tag/v0.2.3) for more details.
*   [2024-08] 🎉🎉 We welcome the new model [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), new tasks [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158). We provide new feature of SGlang Runtime API for llava-onevision model, please refer the [doc](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/commands.md) for inference acceleration
*   [2024-07] 👨‍💻👨‍💻 The `lmms-eval/v0.2.1` has been upgraded to support more models, including [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA), [InternVL-2](https://github.com/OpenGVLab/InternVL), [VILA](https://github.com/NVlabs/VILA), and many more evaluation tasks, e.g. [Details Captions](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/136), [MLVU](https://arxiv.org/abs/2406.04264), [WildVision-Bench](https://huggingface.co/datasets/WildVision/wildvision-arena-data), [VITATECS](https://github.com/lscpku/VITATECS) and [LLaVA-Interleave-Bench](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/).
*   [2024-07] 🎉🎉 We have released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)!
*   [2024-06] 🎬🎬 The `lmms-eval/v0.2.0` has been upgraded to support video evaluations for video models like LLaVA-NeXT Video and Gemini 1.5 Pro across tasks such as EgoSchema, PerceptionTest, VideoMME, and more. Please refer to the [blog](https://lmms-lab.github.io/posts/lmms-eval-0.2/) for more details!
*   [2024-03] 📝📝 We have released the first version of `lmms-eval`, please refer to the [blog](https://lmms-lab.github.io/posts/lmms-eval-0.1/) for more details!

</details>

## Installation

### Using uv (Recommended)

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval

# Create and activate the environment
uv sync
```

### Alternative Installation

```bash
# Create and activate a virtual environment
uv venv eval --python 3.12
source eval/bin/activate

# Install lmms-eval (from Git)
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

<details>
<summary>Reproducing LLaVA-1.5 Results</summary>

Refer to [miscs/repr_scripts.sh](miscs/repr_scripts.sh) and [miscs/repr_torch_envs.txt](miscs/repr_torch_envs.txt) to reproduce the results of LLaVA-1.5.  See [miscs/llava_result_check.md](miscs/llava_result_check.md) for details regarding environment variations.

</details>

If you need to evaluate caption datasets like `coco`, `refcoco`, or `nocaps`, ensure you have `java==1.8.0` installed.

```bash
# Install Java (if needed, using conda)
conda install openjdk=8
```

## Usage

Examples are available in the [examples/models](examples/models) directory.

**Example Evaluation Commands:**

*   **OpenAI-Compatible Model:**  `bash examples/models/openai_compatible.sh` & `bash examples/models/xai_grok.sh`
*   **vLLM:**  `bash examples/models/vllm_qwen2vl.sh`
*   **LLaVA-OneVision:**  `bash examples/models/llava_onevision.sh`
*   **Llama-3-Vision:**  `bash examples/models/llama_vision.sh`
*   **Qwen2-VL:**  `bash examples/models/qwen2_vl.sh` & `bash examples/models/qwen2_5_vl.sh`
*   **LLaVA (on MME):** `bash examples/models/llava_next.sh`

**Additional Parameters:**

```bash
python3 -m lmms_eval --help
```

**Environmental Variables**

Set these variables before running evaluations:

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
```

**Common Issues and Troubleshooting**

Address potential issues related to libraries like `httpx` or `protobuf` by running:

```bash
python3 -m pip install httpx==0.23.3
python3 -m pip install protobuf==3.20
python3 -m pip install numpy==1.26 # If you are using numpy==2.x, sometimes may causing errors
python3 -m pip install sentencepiece # Someties sentencepiece are required for tokenizer to work
```

## Adding Custom Models and Datasets

Refer to the [documentation](docs/README.md) for detailed instructions.

## Acknowledgements

lmms-eval is a fork of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). Review [lm-eval-harness documentation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for comprehensive information.

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

## Contributing

We welcome contributions! Provide feedback, suggest features, or ask questions via issues or pull requests on GitHub.