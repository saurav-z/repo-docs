<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# LMMs-Eval: Your Comprehensive Toolkit for Evaluating Large Multimodal Models

**Quickly and accurately benchmark your LMMs with LMMs-Eval, the leading evaluation suite for text, image, video, and audio tasks.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> LMMs-Eval is a powerful and versatile framework designed to accelerate the development and evaluation of Large Multimodal Models (LMMs) across a wide range of tasks and modalities.

*   🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/)
*   🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab)
*   💬 [Discord Community](https://discord.gg/zdkwKUqrPy)

*   📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md)
*   🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models)
*   📚 [Documentation](docs/README.md)
*   🔗 [Original Repo](https://github.com/EvolvingLMMs-Lab/lmms-eval)

---

## Key Features

*   **Extensive Task Support:** Evaluate your LMMs on a diverse set of text, image, video, and audio tasks, including:
    *   Text Tasks
    *   Image Tasks
    *   Video Tasks
    *   Audio Tasks
*   **Broad Model Compatibility:**  Supports a wide array of LMMs, including popular models like:
    *   LLaVA family
    *   Qwen-VL
    *   OpenAI-compatible models
    *   VLLM integration for accelerated inference
*   **Flexible Installation & Use:** Easy to install and use, with examples and documentation to get you started quickly.
*   **Reproducibility Focus:** Includes scripts and information to reproduce key results, fostering transparency and reliability.
*   **Community Driven:** Benefit from ongoing development, new task & model integration, and community support.
*   **Accelerated Evaluation**: Integrate vLLM and OpenAI API for improved model inference efficiency

---

## What's New

*   **[2025-07]**  🚀🚀 Released `lmms-eval-0.4` with major updates; refer to the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-07]** 🎉🎉 Added support for [PhyX](https://phyx-bench.github.io/).
*   **[2025-06]** 🎉🎉 Added support for [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA).
*   **[2025-04]** 🚀🚀 Introduced [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/).
*   **[2025-02]** 🚀🚀 Integrated [`vllm`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/544) and [`openai_compatible`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/546).

<details>
<summary>Older Updates</summary>

*   [2025-01] 🎓🎓 Released new benchmark: [Video-MMMU](https://arxiv.org/abs/2501.13826).
*   [2024-12] 🎉🎉 Presented [MME-Survey](https://arxiv.org/pdf/2411.15296).
*   [2024-11] 🔈🔊 Added audio evaluations.
*   [2024-10] 🎉🎉 Added support for NaturalBench, TemporalBench, VDC, MovieChat-1K, and Vinoground.
*   [2024-09] 🎉🎉 Added support for MMSearch and MME-RealWorld.
*   [2024-09] ⚙️️⚙️️️️ Upgraded to `0.2.3` with more tasks and features.
*   [2024-08] 🎉🎉 Added support for LLaVA-OneVision, Mantis, MVBench, LongVideoBench, and MMStar.
*   [2024-07] 👨‍💻👨‍💻 Upgraded to `v0.2.1` with more models and evaluation tasks.
*   [2024-07] 🎉🎉 Released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench).
*   [2024-06] 🎬🎬 Upgraded to `v0.2.0` to support video evaluations.
*   [2024-03] 📝📝 Released the first version of `lmms-eval`.
</details>

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

For development:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv venv dev
source dev/bin/activate
uv pip install -e .
```

<details>
<summary>Reproducing LLaVA-1.5 Results</summary>

Use the provided [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) to reproduce the results from the LLaVA-1.5 paper. Check [results check](miscs/llava_result_check.md) for variations.

</details>

If you need to test caption datasets, make sure you have `java==1.8.0` installed by using conda
```
conda install openjdk=8
```
You can check your java version by `java -version`

<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>

Access the detailed results of the LLaVA series models on different datasets and raw data from Weights & Biases:
*   [Google Sheet](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing)
*   [Raw Data](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

</details>

If you want to test [VILA](https://github.com/NVlabs/VILA), you should install the following dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

## Usage Examples

*   [OpenAI-Compatible Models](examples/models/openai_compatible.sh) & [xai_grok.sh](examples/models/xai_grok.sh)
*   [vLLM](examples/models/vllm_qwen2vl.sh)
*   [LLaVA-OneVision](examples/models/llava_onevision.sh)
*   [LLaMA-3.2-Vision](examples/models/llama_vision.sh)
*   [Qwen2-VL](examples/models/qwen2_vl.sh) & [qwen2_5_vl.sh](examples/models/qwen2_5_vl.sh)
*   [LLaVA (MME)](examples/models/llava_next.sh)
*   [Tensor Parallel (llava-next-72b)](examples/models/tensor_parallel.sh)
*   [SGLang (llava-next-72b)](examples/models/sglang.sh)
*   [vLLM (llava-next-72b)](examples/models/vllm_qwen2vl.sh)

**More Parameters:**
```bash
python3 -m lmms_eval --help
```

**Environment Variables:**
Set necessary environment variables (e.g., `OPENAI_API_KEY`, `HF_HOME`, `HF_TOKEN`).

## Adding Custom Models and Datasets

See [documentation](docs/README.md) for how to extend the framework.

## Acknowledgements

LMMs-Eval builds upon the foundation of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

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