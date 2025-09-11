<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# LMMs-Eval: Evaluate and Advance Large Multimodal Models

**Supercharge your LMM development with LMMs-Eval, the comprehensive evaluation suite for text, image, video, and audio tasks!** ([Original Repo](https://github.com/EvolvingLMMs-Lab/lmms-eval))

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

LMMs-Eval is a powerful framework designed to streamline the evaluation of Large Multimodal Models (LMMs), supporting a wide range of tasks across text, image, video, and audio modalities.  Built upon the principles of the highly regarded [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), LMMs-Eval offers a robust and efficient solution for researchers and developers in the rapidly evolving field of multimodal AI.

**Key Features:**

*   **Comprehensive Task Support:** Evaluate LMMs on a vast array of tasks, with support for text, image, video and audio modalities.
*   **Extensive Model Compatibility:**  Ready to evaluate over 30 different models.
*   **Flexible Installation:** Easy installation via `uv` or direct from Git.
*   **Reproducibility:**  Includes scripts and environment details to reproduce results, fostering transparency and reliable research.
*   **OpenAI API Support:**  Seamlessly evaluate models using the OpenAI API format.
*   **Accelerated Evaluation:** Integrations with `vllm` and `SGLang` for faster inference and performance.
*   **Active Development:** Benefit from frequent updates and contributions, with new tasks, models, and features being added regularly.
*   **Detailed Documentation:**  Comprehensive documentation to guide users through setup, usage, and customization ([Documentation](docs/README.md)).
*   **Community Driven:** Engage with a growing community through Discord to share feedback, request features, and contribute to the project.

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Recent Updates

*   **[2025-07]** Major Update with `lmms-eval-0.4`, including new features and improvements.  See the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md) for details.
*   **[2025-07]**  New task: [PhyX](https://phyx-bench.github.io/), a benchmark for physics-grounded reasoning.
*   **[2025-06]**  New task: [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA), for evaluating mathematical reasoning in educational videos.
*   **[2025-04]**  Support for [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/), including batched evaluations.
*   **[2025-02]**  Integration of `vllm` and `openai_compatible` for accelerated evaluation and API-based model support.

<details>
<summary>Previous Updates</summary>

*   **[2025-01]** New benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826).
*   **[2024-12]** Publication of [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296).
*   **[2024-11]** Audio evaluations expanded, supporting Qwen2-Audio and Gemini-Audio.
*   **[2024-10]** Added support for NaturalBench, TemporalBench, VDC, MovieChat-1K, and Vinoground tasks, along with AuroraCap and MovieChat models.
*   **[2024-09]** Added support for MMSearch and MME-RealWorld. Version `0.2.3` released with more tasks and features, including compact language task evaluations and streamlined model/task registration.
*   **[2024-08]** Support for LLaVA-OneVision, Mantis, MVBench, LongVideoBench, and MMStar.  New SGlang Runtime API feature for LLaVA-OneVision.
*   **[2024-07]** Support for LongVA, InternVL-2, VILA, and new evaluation tasks (Details Captions, MLVU, WildVision-Bench, VITATECS, and LLaVA-Interleave-Bench).  Release of technical report and LiveBench.
*   **[2024-06]** Video evaluation support for video models like LLaVA-NeXT Video and Gemini 1.5 Pro.
*   **[2024-03]** Initial release of `lmms-eval`.
</details>

## Installation

### Recommended: Using `uv` for Consistent Environments

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

Refer to the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt).  Check the [results check](miscs/llava_result_check.md) for results in different environments.

</details>

If you need to test on caption datasets, install:
```bash
conda install openjdk=8
```

<details>
<summary>Comprehensive Evaluation Results (LMMs-Eval)</summary>

[Google Sheet with detailed results](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing).  [Raw data exported from Weights & Biases](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).
</details>

If you wish to test [VILA](https://github.com/NVlabs/VILA), please install the following dependencies:
```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```
## Usage Examples

>  More examples are located in [examples/models](examples/models).

**Evaluate OpenAI-Compatible Model:**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Evaluate vLLM:**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Evaluate LLaVA-OneVision:**

```bash
bash examples/models/llava_onevision.sh
```

**Evaluate LLaMA-3-Vision:**

```bash
bash examples/models/llama_vision.sh
```

**Evaluate Qwen2-VL:**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**Evaluate LLaVA on MME:**

```bash
bash examples/models/llava_next.sh
```

**Evaluate with tensor parallel (llava-next-72b):**

```bash
bash examples/models/tensor_parallel.sh
```

**Evaluate with SGLang (llava-next-72b):**

```bash
bash examples/models/sglang.sh
```

**Evaluate with vLLM (llava-next-72b):**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Additional Parameters**

```bash
python3 -m lmms_eval --help
```

## Environment Variables

Set these environment variables before running evaluations:

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include 
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

## Common Issues & Solutions

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26;
python3 -m pip install sentencepiece;
```

## Adding Custom Models and Datasets

Refer to our [documentation](docs/README.md).

## Acknowledgements

LMMs-Eval is based on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

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