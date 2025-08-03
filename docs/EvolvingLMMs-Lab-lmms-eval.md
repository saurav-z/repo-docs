<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# LMMs-Eval: Evaluate and Accelerate Large Multimodal Model Development

**Tired of scattered benchmarks? LMMs-Eval provides a unified framework for comprehensive evaluation of Large Multimodal Models (LMMs), streamlining your research and development.**

[<img src="https://img.shields.io/pypi/v/lmms-eval?style=flat-square" alt="PyPI">](https://pypi.org/project/lmms-eval)
[<img src="https://img.shields.io/pypi/dm/lmms-eval?style=flat-square" alt="Downloads">](https://pypi.org/project/lmms-eval)
[<img src="https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval?style=flat-square" alt="Contributors">](https://github.com/EvolvingLMMs-Lab/lmms-eval)
[<img src="https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval?style=flat-square" alt="Closed Issues">](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[<img src="https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval?style=flat-square" alt="Open Issues">](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features

*   **Comprehensive Evaluation:** Supports a vast array of text, image, video, and audio tasks.
*   **Wide Model Compatibility:**  Evaluates over 30 different LMMs.
*   **Accelerated Evaluation:** Integrated with vLLM and supports OpenAI API compatible models for faster results.
*   **Easy to Use:**  Simple installation and clear usage examples.
*   **Actively Maintained:**  Regular updates with new tasks, models, and features.
*   **Community Driven:** Open to contributions and feedback from the research community.
*   **Reproducibility:**  Includes scripts and instructions for reproducing paper results.

## Recent Updates & Announcements

*   **[2025-07]**  Released `lmms-eval-0.4` with major updates. See [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-07]**  New task: [PhyX](https://phyx-bench.github.io/).
*   **[2025-06]**  New task: [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA).
*   **[2025-04]**  Support for Aero-1-Audio (batched evaluation).
*   **[2025-02]**  Integrated `vllm` and `openai_compatible` for faster and broader model support.

<details>
<summary>Show more updates</summary>

*   [2025-01] 🎓🎓 Released benchmark: [Video-MMMU](https://arxiv.org/abs/2501.13826).
*   [2024-12] 🎉🎉 Presented [MME-Survey](https://arxiv.org/pdf/2411.15296).
*   [2024-11] 🔈🔊 Added audio evaluations for Qwen2-Audio and Gemini-Audio.
*   [2024-10] 🎉🎉 New tasks: [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench), [VDC](https://rese1f.github.io/aurora-web/), [MovieChat-1K](https://rese1f.github.io/MovieChat/), and [Vinoground](https://vinoground.github.io/). New models: [AuroraCap](https://github.com/rese1f/aurora) and [MovieChat](https://rese1f/MovieChat).
*   [2024-09] 🎉🎉 New tasks: [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/).
*   [2024-09] ⚙️️⚙️️️️ Upgraded to `0.2.3` with language tasks and reduced overhead.
*   [2024-08] 🎉🎉 New models: [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), new tasks [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158).
*   [2024-07] 👨‍💻👨‍💻 Upgraded to `lmms-eval/v0.2.1` with more models and tasks.
*   [2024-07] 🎉🎉 Released [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)!
*   [2024-06] 🎬🎬 Upgraded to `lmms-eval/v0.2.0` with video evaluations.
*   [2024-03] 📝📝 Released the first version of `lmms-eval`.

</details>

---

## Installation

**Installation Instructions:**

```bash
# Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

# For development
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv venv dev
source dev/bin/activate
uv pip install -e .
```

<details>
<summary>Reproducing LLaVA-1.5 Results</summary>
**Reproducing LLaVA-1.5 results requires the following:**
*   Install Java: `conda install openjdk=8` and check your java version: `java -version`
*   Check the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt)
*   Review the [results check](miscs/llava_result_check.md).
</details>

## Usage Examples

```bash
# Evaluate OpenAI-Compatible Model
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh

# Evaluate vLLM
bash examples/models/vllm_qwen2vl.sh

# Evaluate LLaVA-OneVision
bash examples/models/llava_onevision.sh

# Evaluate LLaMA-3.2-Vision
bash examples/models/llama_vision.sh

# Evaluate Qwen2-VL
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh

# Evaluate LLaVA
bash examples/models/llava_next.sh

# Evaluate with tensor parallel
bash examples/models/tensor_parallel.sh

# Evaluate with SGLang
bash examples/models/sglang.sh

# Evaluate with vLLM
bash examples/models/vllm_qwen2vl.sh

# Get more parameters
python3 -m lmms_eval --help
```

## Environment Variables

**Essential Environment Variables:**

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
```

## Add Customized Model and Dataset

Please refer to our [documentation](docs/README.md) to add your custom model and dataset

## Acknowledgements

This project is built upon the foundational work of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), and you can review their [docs](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs).

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

[Back to Top](#lmms-eval-evaluate-and-accelerate-large-multimodal-model-development)