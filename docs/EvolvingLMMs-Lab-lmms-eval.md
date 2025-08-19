<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png" alt="lmms-eval logo">
</p>

# lmms-eval: The Comprehensive Evaluation Suite for Large Multimodal Models

**Evaluate and accelerate the development of your large multimodal models with `lmms-eval`, supporting a wide range of tasks and models.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md) | [Original Repository](https://github.com/EvolvingLMMs-Lab/lmms-eval)

---

## Key Features

*   **Extensive Task Coverage:** Support for a vast array of text, image, video, and audio tasks.
*   **Broad Model Compatibility:** Compatible with over 30 large multimodal models.
*   **Accelerated Evaluation:** Integrated with `vllm` for faster evaluation and support for OpenAI-compatible API models.
*   **Reproducibility:** Provides environment scripts and result checks for LLaVA-1.5 paper results.
*   **Customization:**  Easily add your own models and datasets.
*   **Regular Updates:** Continuously updated with new tasks, models, and features.

---

## Recent Updates & Announcements

*   **[2025-07]** 🚀🚀 Released `lmms-eval-0.4` with major updates; see the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-07]** 🎉🎉 Added support for [PhyX](https://phyx-bench.github.io/) benchmark.
*   **[2025-06]** 🎉🎉 Integrated [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA) evaluation.
*   **[2025-04]** 🚀🚀 Introduced Aero-1-Audio with batched evaluations.
*   **[2025-02]** 🚀🚀 Integrated `vllm` and `openai_compatible` for accelerated and flexible model evaluations.

<details>
<summary>More Updates</summary>

*   **[2025-01]** 🎓🎓 Released [Video-MMMU](https://arxiv.org/abs/2501.13826) benchmark.
*   **[2024-12]** 🎉🎉 Presented [MME-Survey](https://arxiv.org/pdf/2411.15296).
*   **[2024-11]** 🔈🔊 Added audio evaluations.
*   **[2024-10]** 🎉🎉 Support for [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench), [VDC](https://rese1f.github.io/aurora-web/), [MovieChat-1K](https://rese1f.github.io/MovieChat/), and [Vinoground](https://vinoground.github.io/).
*   **[2024-09]** 🎉🎉 Added [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/) support.
*   **[2024-09]** ⚙️️⚙️️️️ Updated `lmms-eval` to `0.2.3`.
*   **[2024-08]** 🎉🎉 Added support for [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158) and SGlang Runtime API.
*   **[2024-07]** 👨‍💻👨‍💻 Updated `lmms-eval/v0.2.1`, adding more models, like [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA), [InternVL-2](https://github.com/OpenGVLab/InternVL), [VILA](https://github.com/NVlabs/VILA), and tasks like [Details Captions](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/136), [MLVU](https://arxiv.org/abs/2406.04264), [WildVision-Bench](https://huggingface.co/datasets/WildVision/wildvision-arena-data), [VITATECS](https://github.com/lscpku/VITATECS) and [LLaVA-Interleave-Bench](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/).
*   **[2024-07]** 🎉🎉 Released [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench).
*   **[2024-06]** 🎬🎬 Support for video evaluations.
*   **[2024-03]** 📝📝 Released the first version of `lmms-eval`.

</details>

---

## Installation

Install `lmms-eval` easily using `uv` or by cloning the repository. Detailed instructions are provided for both methods.

### Using `uv` (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

### For Development (Cloning)

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv venv dev
source dev/bin/activate
uv pip install -e .
```

### Additional Dependencies

If you plan to evaluate caption datasets like `coco`, `refcoco`, or `nocaps`, you'll need `java==1.8.0`. Install it with:

```bash
conda install openjdk=8
```

Then, verify your Java version:

```bash
java -version
```

### Reproducing LLaVA-1.5 Results

Follow the instructions in `miscs/repr_scripts.sh` and check `miscs/repr_torch_envs.txt` to reproduce the LLaVA-1.5 paper results.  Consult `miscs/llava_result_check.md` for result comparisons across different environments.

---

## Usage Examples

Explore a variety of usage examples in the `examples/models` directory to get started.

*   **OpenAI-Compatible Models:**
    ```bash
    bash examples/models/openai_compatible.sh
    bash examples/models/xai_grok.sh
    ```
*   **vLLM Evaluation:**
    ```bash
    bash examples/models/vllm_qwen2vl.sh
    ```
*   **LLaVA-OneVision:**
    ```bash
    bash examples/models/llava_onevision.sh
    ```
*   **LLaMA-3.2-Vision:**
    ```bash
    bash examples/models/llama_vision.sh
    ```
*   **Qwen2-VL:**
    ```bash
    bash examples/models/qwen2_vl.sh
    bash examples/models/qwen2_5_vl.sh
    ```
*   **LLaVA (on MME):**
    If you want to test LLaVA 1.5, you will have to clone their repo from [LLaVA](https://github.com/haotian-liu/LLaVA) and
    ```bash
    bash examples/models/llava_next.sh
    ```
*   **Tensor Parallel Evaluation:**
    ```bash
    bash examples/models/tensor_parallel.sh
    ```
*   **SGLang Evaluation:**
    ```bash
    bash examples/models/sglang.sh
    ```
*   **vLLM for Qwen2VL:**
    ```bash
    bash examples/models/vllm_qwen2vl.sh
    ```

Explore the command line options with:

```bash
python3 -m lmms_eval --help
```

### Environment Variables

Set these environment variables for optimal performance:

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

### Addressing Common Issues

Resolve issues related to `httpx` or `protobuf` by running:

```bash
python3 -m pip install httpx==0.23.3
python3 -m pip install protobuf==3.20
python3 -m pip install numpy==1.26  # If using numpy==2.x causing errors
python3 -m pip install sentencepiece  # If tokenizer needs it
```

---

## Contributing and Documentation

Contribute to `lmms-eval` by providing feedback and feature suggestions through issues or pull requests. Consult the [documentation](docs/README.md) for details on adding custom models and datasets.

## Acknowledgements

`lmms-eval` is built upon the foundation of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

---

## Citations

```bibtex
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