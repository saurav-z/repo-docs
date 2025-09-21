<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png" alt="LMMs-Eval Logo">
</p>

# LMMs-Eval: The Comprehensive Evaluation Suite for Large Multimodal Models

**Unleash the potential of Large Multimodal Models (LMMs) with LMMs-Eval, your go-to framework for rigorous and efficient evaluation.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> **[Original Repository](https://github.com/EvolvingLMMs-Lab/lmms-eval)**

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features

*   **Extensive Task Support:** Evaluate your LMMs across a wide range of text, image, video, and audio tasks.
*   **Model Compatibility:** Supports over 30 different LMMs.
*   **Efficient Evaluation:** Built for consistent and efficient evaluation of LMMs.
*   **Reproducibility:** Provides environment setup scripts and result checks for reliable results.
*   **Community-Driven:** Benefit from ongoing development and contributions.
*   **Integration with Advanced Technologies:** Includes vLLM and OpenAI API compatibility for accelerated and versatile evaluations.

---

## Recent Updates & Announcements

*   **[2025-07]** `lmms-eval-0.4` released with major updates; See [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-07]** New task: [PhyX](https://phyx-bench.github.io/), a benchmark for physics-grounded reasoning.
*   **[2025-06]** New task: [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA), for evaluating mathematical reasoning in videos.
*   **[2025-04]** Support for [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/), with batched evaluation capabilities.
*   **[2025-02]** Integrated `vllm` and `openai_compatible` for faster evaluation and API-based model support.

<details>
<summary>Older Updates</summary>

-   [2025-01] 🎓🎓 Released new benchmark: [Video-MMMU](https://arxiv.org/abs/2501.13826).
-   [2024-12] 🎉🎉 Presented [MME-Survey](https://arxiv.org/pdf/2411.15296), a comprehensive survey.
-   [2024-11] 🔈🔊 `lmms-eval/v0.3.0` now supports audio evaluations.
-   [2024-10] 🎉🎉 Added NaturalBench, TemporalBench, VDC, MovieChat-1K, and Vinoground benchmarks, along with support for AuroraCap and MovieChat models.
-   [2024-09] 🎉🎉 Added MMSearch and MME-RealWorld benchmarks. `lmms-eval` upgraded to `0.2.3`.
-   [2024-08] 🎉🎉 New model LLaVA-OneVision, Mantis, and new benchmarks MVBench, LongVideoBench, MMStar. SGlang Runtime API support for llava-onevision.
-   [2024-07] 👨‍💻👨‍💻 `lmms-eval/v0.2.1` upgraded to support more models, including LongVA, InternVL-2, VILA, and many more evaluation tasks.
-   [2024-07] 🎉🎉 Released the technical report and LiveBench.
-   [2024-06] 🎬🎬 `lmms-eval/v0.2.0` upgraded to support video evaluations.
-   [2024-03] 📝📝 Released the first version of `lmms-eval`.

</details>

---

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

<details>
<summary>Reproducing LLaVA-1.5 Results</summary>

See [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) for reproducing LLaVA-1.5 results.  Check [results check](miscs/llava_result_check.md) for results variations across different environments.

</details>

For caption dataset testing (`coco`, `refcoco`, `nocaps`), install `java==1.8.0` (using `conda install openjdk=8`).

<details>
<summary>Comprehensive Evaluation Results</summary>

Access the detailed results of LLaVA series models on different datasets:

*   Google Sheet: [Link to Google Sheet](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing)
*   Raw data (Weights & Biases): [Link to Raw Data](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing)

</details>

To test [VILA](https://github.com/NVlabs/VILA):

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

---

## Usages

> More examples can be found in [examples/models](examples/models)

**Examples:**

*   Evaluation of OpenAI-Compatible Model: `bash examples/models/openai_compatible.sh`
*   Evaluation of vLLM: `bash examples/models/vllm_qwen2vl.sh`
*   Evaluation of LLaVA-OneVision: `bash examples/models/llava_onevision.sh`
*   ... (and more, as shown in the original README)

**More Parameters:**

```bash
python3 -m lmms_eval --help
```

**Environmental Variables**
```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
```
**Common Environment Issues**
```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26;
python3 -m pip install sentencepiece;
```

---

## Contributing

We encourage contributions!  Provide feedback, suggest features, or submit pull requests via GitHub.

## Add Customized Model and Dataset

Please refer to our [documentation](docs/README.md).

## Acknowledgement

`lmms_eval` is a fork of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). Read the [lm-eval-harness docs](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for related information.

---

### Citations

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