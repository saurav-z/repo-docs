<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png" alt="LMMs-Eval Logo">
</p>

# LMMs-Eval: The Comprehensive Evaluation Suite for Large Multimodal Models

> **Tackle the complexity of evaluating cutting-edge multimodal models with LMMs-Eval, your go-to toolkit for standardized benchmarking and analysis.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

**[View the Original Repository](https://github.com/EvolvingLMMs-Lab/lmms-eval)**

LMMs-Eval empowers researchers and developers to rigorously evaluate large multimodal models (LMMs) across various tasks, providing a robust framework for advancing AI capabilities.

**Key Features:**

*   **Extensive Task Support:** Evaluates LMMs on a wide range of text, image, video, and audio tasks. [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md)
*   **Model Compatibility:** Supports over 30 different LMMs, including leading models from various research groups. [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models)
*   **Flexible Installation:** Provides installation methods using `uv` for environment consistency and direct installation via Git.
*   **Reproducibility:** Includes scripts and resources to reproduce results from key LMM papers.
*   **Accelerated Evaluation:** Integrates with vLLM and supports OpenAI API-compatible models for faster evaluation.
*   **Comprehensive Documentation:** Detailed documentation and examples to guide users through the evaluation process. [Documentation](docs/README.md)
*   **Active Community:**  Engage with the community via [discord/lmms-eval](https://discord.gg/zdkwKUqrPy) for support, feedback, and contributions.
*   **Regular Updates:** Stay up-to-date with the latest benchmarks, models, and features.

**Announcements:**

*   **[2025-07]**  🚀🚀 Released `lmms-eval-0.4` with major updates, including new features and improvements. See the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-07]** 🎉🎉 Welcoming the new task [PhyX](https://phyx-bench.github.io/), a physics-grounded reasoning benchmark for visual scenarios.
*   **[2025-06]** 🎉🎉 Welcoming the new task [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA), for evaluating mathematical reasoning in educational videos.
*   **[2025-04]** 🚀🚀 Introducing [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/), an audio model with batched evaluation support.
*   **[2025-02]** 🚀🚀 Integrated [`vllm`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/544) and [`openai_compatible`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/546) for accelerated and API-based model evaluations.

<details>
<summary>See past updates</summary>

-   [2025-01] 🎓🎓 Released new benchmark: [Video-MMMU](https://arxiv.org/abs/2501.13826).
-   [2024-12] 🎉🎉 Presented [MME-Survey](https://arxiv.org/pdf/2411.15296).
-   [2024-11] 🔈🔊 Audio evaluations now supported. See [blog](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.3.md).
-   [2024-10] 🎉🎉 Added [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench).
-   [2024-10] 🎉🎉 Added tasks [VDC](https://rese1f.github.io/aurora-web/), [MovieChat-1K](https://rese1f.github.io/MovieChat/), and [Vinoground](https://vinoground.github.io/), and models [AuroraCap](https://github.com/rese1f/aurora) and [MovieChat](https://rese1f/MovieChat).
-   [2024-09] 🎉🎉 Added [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/).
-   [2024-09] ⚙️️️️️ Upgraded to `0.2.3`. See [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/releases/tag/v0.2.3).
-   [2024-08] 🎉🎉 Added [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), and [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158).
-   [2024-07] 👨‍💻👨‍💻 Upgraded to `0.2.1`. See [doc](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/commands.md).
-   [2024-07] 🎉🎉 Released [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)!
-   [2024-06] 🎬🎬 Upgraded to `0.2.0`. See [blog](https://lmms-lab.github.io/posts/lmms-eval-0.2/).
-   [2024-03] 📝📝 Released the first version of `lmms-eval`. See [blog](https://lmms-lab.github.io/posts/lmms-eval-0.1/).
</details>

## Installation

### Recommended: Using `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv sync
```

```bash
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

See [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt). Results check available [here](miscs/llava_result_check.md).
</details>

```bash
conda install openjdk=8
```

<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>

[Detailed results](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing) and [raw data](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).
</details>

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

## Usage Examples

> Find more examples in [examples/models](examples/models)

**OpenAI-Compatible Models**

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

**Llama-3.2-Vision Evaluation**

```bash
bash examples/models/llama_vision.sh
```

**Qwen2-VL Evaluation**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**LLaVA on MME**

```bash
bash examples/models/llava_next.sh
```

**Tensor Parallel Evaluation**

```bash
bash examples/models/tensor_parallel.sh
```

**SGLang Evaluation**

```bash
bash examples/models/sglang.sh
```

**vLLM for Bigger Models**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**More Parameters**

```bash
python3 -m lmms_eval --help
```

## Environment Variables

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
```

## Common Issues

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26;
python3 -m pip install sentencepiece;
```

## Adding Custom Models and Datasets

Refer to the [documentation](docs/README.md).

## Acknowledgements

Based on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

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