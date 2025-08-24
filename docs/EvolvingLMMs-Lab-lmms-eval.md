<p align="center" width="70%">
<img src="https://i.postimg.com/KvkLzbF9/WX20241212-014400-2x.png" alt="lmms-eval Logo">
</p>

# lmms-eval: The Comprehensive Evaluation Suite for Large Multimodal Models

**Evaluate and compare large multimodal models (LMMs) efficiently and comprehensively with lmms-eval, your go-to framework for advancing multimodal AI.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

>  Accelerating the development of large multimodal models (LMMs) with `lmms-eval`. We support most text, image, video and audio tasks.

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features of lmms-eval

*   **Extensive Task Support:** Evaluate LMMs across a wide range of tasks including text, image, video, and audio.
*   **Broad Model Compatibility:**  Supports over 30 different LMMs.
*   **Efficient Evaluation:** Designed for consistent and rapid evaluation.
*   **Flexible Installation:** Supports `uv` for environment consistency, and alternative installation methods.
*   **Reproducibility:** Includes scripts and resources to reproduce LLaVA-1.5 paper results.
*   **OpenAI API Support:** Enables evaluation of models that follow the OpenAI API format.
*   **VLLM Integration:**  Accelerates evaluation using vLLM.
*   **Continuous Updates:**  Regularly updated with new tasks, models, and features.

## What's New

*   **[2025-07] lmms-eval-0.4 Release:** Major update with new features and improvements. Refer to the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md) for details.
*   **[2025-07] New Benchmark: PhyX:** Assess models' physics-grounded reasoning in visual scenarios.
*   **[2025-06] New Benchmark: VideoMathQA:** Evaluate mathematical reasoning in educational videos.
*   **[2025-04] New Model Support: Aero-1-Audio:** Evaluate this compact audio model with batched evaluations.
*   **[2025-02] vLLM and OpenAI API Integration:**  Accelerated evaluation and support for OpenAI-compatible models.
*   **[2025-01] New Benchmark: Video-MMMU:** Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos.
*   **[2024-12] MME-Survey:** A Comprehensive Survey on Evaluation of Multimodal LLMs
*   **[2024-11] Audio Evaluation:** Added support for audio models like Qwen2-Audio and Gemini-Audio.
*   **[2024-10] NaturalBench:** New vision-centric VQA benchmark.
*   **[2024-10] TemporalBench:** New benchmark for fine-grained temporal understanding and reasoning for videos.
*   **[2024-10] New Tasks:** VDC, MovieChat-1K, and Vinoground.
*   **[2024-10] New Models:** AuroraCap and MovieChat.
*   **[2024-09] MMSearch and MME-RealWorld:** for inference acceleration.
*   **[2024-09] lmms-eval 0.2.3 Release:** with more tasks and features.
*   **[2024-08] New Models and Tasks:** LLaVA-OneVision, Mantis, MVBench, LongVideoBench, MMStar.
*   **[2024-07] More Model Support:** LongVA, InternVL-2, VILA.
*   **[2024-07] New Tasks:** Details Captions, MLVU, WildVision-Bench, VITATECS, LLaVA-Interleave-Bench.
*   **[2024-07] Technical Report and LiveBench Release.**
*   **[2024-06] Video Evaluation Support:** for video models like LLaVA-NeXT Video and Gemini 1.5 Pro.
*   **[2024-03] Initial Release:** The first version of `lmms-eval`.

<details>
<summary>Below is a chronological list of recent tasks, models, and features added by our amazing contributors. </summary>

- [2025-01] 🎓🎓 We have released our new benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826). Please refer to the [project page](https://videommmu.github.io/) for more details.
- [2024-12] 🎉🎉 We have presented [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296), jointly with [MME Team](https://github.com/BradyFU/Video-MME) and [OpenCompass Team](https://github.com/open-compass).
- [2024-11] 🔈🔊 The `lmms-eval/v0.3.0` has been upgraded to support audio evaluations for audio models like Qwen2-Audio and Gemini-Audio across tasks such as AIR-Bench, Clotho-AQA, LibriSpeech, and more. Please refer to the [blog](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.3.md) for more details!
- [2024-10] 🎉🎉 We welcome the new task [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), a vision-centric VQA benchmark (NeurIPS'24) that challenges vision-language models with simple questions about natural imagery.
- [2024-10] 🎉🎉 We welcome the new task [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench) for fine-grained temporal understanding and reasoning for videos, which reveals a huge (>30%) human-AI gap.
- [2024-10] 🎉🎉 We welcome the new tasks [VDC](https://rese1f.github.io/aurora-web/) for video detailed captioning, [MovieChat-1K](https://rese1f.github.io/MovieChat/) for long-form video understanding, and [Vinoground](https://vinoground.github.io/), a temporal counterfactual LMM benchmark composed of 1000 short natural video-caption pairs. We also welcome the new models: [AuroraCap](https://github.com/rese1f/aurora) and [MovieChat](https://rese1f/MovieChat).
- [2024-09] 🎉🎉 We welcome the new tasks [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/) for inference acceleration
- [2024-09] ⚙️️⚙️️️️ We upgrade `lmms-eval` to `0.2.3` with more tasks and features. We support a compact set of language tasks evaluations (code credit to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)), and we remove the registration logic at start (for all models and tasks) to reduce the overhead. Now `lmms-eval` only launches necessary tasks/models. Please check the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/releases/tag/v0.2.3) for more details.
- [2024-08] 🎉🎉 We welcome the new model [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), new tasks [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158). We provide new feature of SGlang Runtime API for llava-onevision model, please refer the [doc](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/commands.md) for inference acceleration
- [2024-07] 👨‍💻👨‍💻 The `lmms-eval/v0.2.1` has been upgraded to support more models, including [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA), [InternVL-2](https://github.com/OpenGVLab/InternVL), [VILA](https://github.com/NVlabs/VILA), and many more evaluation tasks, e.g. [Details Captions](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/136), [MLVU](https://arxiv.org/abs/2406.04264), [WildVision-Bench](https://huggingface.co/datasets/WildVision/wildvision-arena-data), [VITATECS](https://github.com/lscpku/VITATECS) and [LLaVA-Interleave-Bench](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/).
- [2024-07] 🎉🎉 We have released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)! 
- [2024-06] 🎬🎬 The `lmms-eval/v0.2.0` has been upgraded to support video evaluations for video models like LLaVA-NeXT Video and Gemini 1.5 Pro across tasks such as EgoSchema, PerceptionTest, VideoMME, and more. Please refer to the [blog](https://lmms-lab.github.io/posts/lmms-eval-0.2/) for more details!
- [2024-03] 📝📝 We have released the first version of `lmms-eval`, please refer to the [blog](https://lmms-lab.github.io/posts/lmms-eval-0.1/) for more details!

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
<summary>Reproducing LLaVA-1.5 Results</summary>
<br>
Check the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) to reproduce LLaVA-1.5 paper results.  Refer to [results check](miscs/llava_result_check.md) for variations.
</details>

If you need to test caption datasets like `coco`, `refcoco`, and `nocaps`, you will need `java==1.8.0`. If you don't have it, install with conda:
```bash
conda install openjdk=8
```
Verify the installation with `java -version`.

<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>
<br>

Detailed dataset information and results for the LLaVA series models are available in this [Google Sheet](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing).  Raw data from Weights & Biases is available [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).
</details>

If you want to test [VILA](https://github.com/NVlabs/VILA), you should install the following dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

## Usage Examples

> More examples can be found in the [examples/models](examples/models) directory.

**OpenAI-Compatible Model Evaluation**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**vLLM Evaluation**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**LLaVA-OneVision Evaluation**

```bash
bash examples/models/llava_onevision.sh
```

**LLaMA-3.2-Vision Evaluation**

```bash
bash examples/models/llama_vision.sh
```

**Qwen2-VL Evaluation**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**LLaVA on MME Evaluation**

```bash
bash examples/models/llava_next.sh
```

**Tensor Parallel Evaluation (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**SGLang Evaluation (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**vLLM Evaluation (llava-next-72b)**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Additional Parameters**

```bash
python3 -m lmms_eval --help
```

**Environment Variables**

Set these environment variables before running:

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

Resolve issues related to httpx or protobuf by:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26; # If you are using numpy==2.x, sometimes may causing errors
python3 -m pip install sentencepiece; # Someties sentencepiece are required for tokenizer to work
```

## Adding Custom Models and Datasets

Please refer to the [documentation](docs/README.md) for instructions.

## Acknowledgements

`lmms-eval` is based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) project. Consult the [lm-evaluation-harness documentation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for related information.

---

## Changes from the Original API

*   Build context now passes in `idx` and processes images/docs during model response.
*   `Instance.args` now contains a list of images.
*   Separate model classes are used for each LMM due to input/output format differences (future unification planned).

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

**[Contribute to the project](https://github.com/EvolvingLMMs-Lab/lmms-eval)**