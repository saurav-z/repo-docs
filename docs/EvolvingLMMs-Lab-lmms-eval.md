<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png" alt="LMMs-Eval Logo">
</p>

# LMMs-Eval: The Comprehensive Evaluation Suite for Large Multimodal Models

**Accelerate your research in Large Multimodal Models (LMMs) with LMMs-Eval, your one-stop solution for consistent and efficient LMM evaluation.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

[**View the original repository on GitHub**](https://github.com/EvolvingLMMs-Lab/lmms-eval)

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features of LMMs-Eval:

*   **Extensive Task Support:** Evaluate LMMs across text, image, video, and audio tasks.
*   **Broad Model Compatibility:** Supports a wide range of LMMs, including popular architectures like LLaVA, Qwen-VL, and more.
*   **Efficient Evaluation:** Optimized for speed and consistency in evaluating LMMs.
*   **Easy Installation:** Streamlined installation process using `uv` for reproducible environments, with alternative installation options available.
*   **Reproducibility Focus:** Provides scripts and resources to help reproduce results, promoting transparency and scientific rigor.
*   **Regular Updates:** Continuously updated with new tasks, models, and features to stay at the forefront of LMM research.

---

## Recent Updates and Announcements

*   **[2025-07]** 🚀🚀 Released `lmms-eval-0.4` with significant updates; see [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-07]** 🎉🎉 Support for [PhyX](https://phyx-bench.github.io/), a physics-grounded reasoning benchmark.
*   **[2025-06]** 🎉🎉 Support for [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA), focusing on mathematical reasoning in videos.
*   **[2025-04]** 🚀🚀 Introduced [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/) and added support for batched evaluations.
*   **[2025-02]** 🚀🚀 Integrated `vllm` for accelerated evaluation and `openai_compatible` for API-based models.

<details>
<summary>More Recent Updates</summary>
    *   [2025-01] 🎓🎓 Released new benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826).
    *   [2024-12] 🎉🎉 Presented [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296).
    *   [2024-11] 🔈🔊 `lmms-eval/v0.3.0` supports audio evaluations for models such as Qwen2-Audio and Gemini-Audio.
    *   [2024-10] 🎉🎉 Support for NaturalBench, TemporalBench, VDC, MovieChat-1K, and Vinoground.
    *   [2024-09] 🎉🎉 Support for MMSearch and MME-RealWorld.
    *   [2024-09] ⚙️️⚙️️️️ Upgraded `lmms-eval` to `0.2.3` with more tasks, features, and overhead reduction.
    *   [2024-08] 🎉🎉 Support for LLaVA-OneVision, Mantis, MVBench, LongVideoBench, and MMStar.
    *   [2024-07] 👨‍💻👨‍💻 Upgraded `lmms-eval` to `v0.2.1` with support for LongVA, InternVL-2, VILA, and more evaluation tasks.
    *   [2024-07] 🎉🎉 Released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench).
    *   [2024-06] 🎬🎬 Upgraded `lmms-eval` to `v0.2.0` with video evaluations for video models.
    *   [2024-03] 📝📝 Released the first version of `lmms-eval`.
</details>

---

## Installation

### Using uv (Recommended)

Install `uv` for consistent package versions:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Set up a development environment:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv sync  # Creates/updates environment
```

Run commands:

```bash
uv run python -m lmms_eval --help  # Run commands using uv run
```

Add dependencies:

```bash
uv add <package>  # Updates both pyproject.toml and uv.lock
```

### Alternative Installation

Direct installation from Git:

```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

<details>
<summary>Reproducing LLaVA-1.5 Paper Results</summary>

Check the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) to reproduce LLaVA-1.5's results. Refer to [results check](miscs/llava_result_check.md) for potential result variations.

</details>

If you need to test caption datasets like `coco`, `refcoco`, and `nocaps`, install `java==1.8.0`:

```bash
conda install openjdk=8
```

Check your Java version:

```bash
java -version
```

<details>
<summary>Comprehensive Evaluation Results</summary>

A detailed Google Sheet of the LLaVA series model results on different datasets can be accessed [here](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing).
Raw data exported from Weights & Biases is available [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

</details>
<br>

If you want to test [VILA](https://github.com/NVlabs/VILA), you should install the following dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

We welcome your feedback, feature requests, and contributions. Please submit issues or pull requests on GitHub.

---

## Usage Examples

> More examples can be found in [examples/models](examples/models)

**Evaluate OpenAI-Compatible Model**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Evaluate vLLM**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Evaluate LLaVA-OneVision**

```bash
bash examples/models/llava_onevision.sh
```

**Evaluate LLaMA-3.2-Vision**

```bash
bash examples/models/llama_vision.sh
```

**Evaluate Qwen2-VL**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**Evaluate LLaVA on MME**

If you want to test LLaVA 1.5, you will have to clone their repo from [LLaVA](https://github.com/haotian-liu/LLaVA) and

```bash
bash examples/models/llava_next.sh
```

**Evaluate with tensor parallel for bigger model (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**Evaluate with SGLang for bigger model (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**Evaluate with vLLM for bigger model (llava-next-72b)**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**More Parameters**

```bash
python3 -m lmms_eval --help
```

**Environmental Variables**

Before running, set the following environment variables:

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

Address common issues by trying:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# If you are using numpy==2.x, sometimes may causing errors
python3 -m pip install numpy==1.26;
# Someties sentencepiece are required for tokenizer to work
python3 -m pip install sentencepiece;
```

---

## Adding Customized Models and Datasets

Refer to our [documentation](docs/README.md) for details.

---

## Acknowledgements

LMMs-Eval is built upon [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). Consult the [lm-eval-harness documentation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for related information.

---

## Changes from the Original API:

*   Context building now only processes the index and image during the model responding phase.
*   `Instance.args` contains a list of images for input.
*   The project addresses the limitations of HF language models and adapts to create a new class for each lmms model.

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