<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png" alt="lmms-eval Logo">
</p>

# LMMs-Eval: Your Comprehensive Suite for Evaluating Large Multimodal Models

>  **LMMs-Eval** is a powerful evaluation framework designed to accelerate the development of Large Multimodal Models (LMMs).

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

**Explore the cutting edge of multimodal AI with LMMs-Eval!**  This robust framework provides everything you need to benchmark and analyze LMMs across various modalities.

[**Visit the original repository on GitHub**](https://github.com/EvolvingLMMs-Lab/lmms-eval)

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features

*   **Comprehensive Task Support:**  Evaluate LMMs across text, image, video, and audio tasks.
*   **Wide Model Compatibility:**  Supports over 30 different LMMs, with new models added frequently.
*   **Accelerated Evaluation:** Integrates with `vllm` for faster model evaluation.
*   **OpenAI API Support:**  Evaluate any API-based model compatible with the OpenAI API format.
*   **Reproducibility:** Provides scripts and resources to reproduce paper results.
*   **Flexible Usage:** Includes detailed examples and documentation for easy integration.
*   **Community Driven:** Actively maintained and updated with contributions from a vibrant community.

---

## Recent Updates and Announcements

*   **[2025-07]** 🚀🚀 Release of `lmms-eval-0.4` with major updates and improvements. See [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-04]** 🚀🚀 Support for Aero-1-Audio, including batched evaluations.
*   **[2025-07]** 🎉🎉 Introduction of the new task [PhyX](https://phyx-bench.github.io/).
*   **[2025-06]** 🎉🎉 Introduction of the new task [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA).
*   **[2025-02]** 🚀🚀 Integration of `vllm` and `openai_compatible` for faster and broader model support.

<details>
<summary>More Recent Updates</summary>

*   **[2025-01]** 🎓🎓 Released the new benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826).
*   **[2024-12]** 🎉🎉 Presented [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296).
*   **[2024-11]** 🔈🔊  Audio evaluation support for audio models like Qwen2-Audio and Gemini-Audio.
*   **[2024-10]** 🎉🎉 New Tasks: [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench), [VDC](https://rese1f.github.io/aurora-web/), [MovieChat-1K](https://rese1f.github.io/MovieChat/), and [Vinoground](https://vinoground.github.io/).  New Models: [AuroraCap](https://github.com/rese1f/aurora) and [MovieChat](https://github.com/rese1f/MovieChat).
*   **[2024-09]** 🎉🎉 New tasks [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/) for inference acceleration.
*   **[2024-09]** ⚙️️⚙️️️️ Upgrade `lmms-eval` to `0.2.3`.  See the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/releases/tag/v0.2.3).
*   **[2024-08]** 🎉🎉 New models [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), and new tasks [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158).
*   **[2024-07]** 👨‍💻👨‍💻 Upgraded `lmms-eval/v0.2.1` to support more models and evaluation tasks.
*   **[2024-07]** 🎉🎉 Released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench).
*   **[2024-06]** 🎬🎬 Upgraded `lmms-eval/v0.2.0` to support video evaluations.
*   **[2024-03]** 📝📝 Released the first version of `lmms-eval`.

</details>

---

## Installation

### Install with `uv` (recommended)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

### Development Installation
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv venv dev
source dev/bin/activate
uv pip install -e .
```

<details>
<summary>Reproducing LLaVA-1.5 Results</summary>

Follow the instructions in `miscs/repr_scripts.sh` and review `miscs/repr_torch_envs.txt` to reproduce LLaVA-1.5 results.  Check `miscs/llava_result_check.md` for results in different environments.

</details>

**Dependencies:**

*   Install `java==1.8.0` to work with `pycocoeval` (for caption datasets like `coco`, `refcoco`, and `nocaps`)

```bash
conda install openjdk=8
```

*   Test [VILA](https://github.com/NVlabs/VILA) requires

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

---

## Evaluation Results

<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>
<br>
<p>
  Detailed information is available in the Google Sheet, including datasets and specifics.  See the live sheet [here](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing).
</p>

<p align="center" width="100%">
<img src="https://i.postimg.cc/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%">
</p>

<p>
  Raw data exported from Weights & Biases is available [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).
</p>
</details>
<br>

---

## Usage Examples

> See more examples in [examples/models](examples/models)

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

**Llama-3.2-Vision Evaluation:**

```bash
bash examples/models/llama_vision.sh
```

**Qwen2-VL Evaluation:**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**LLaVA on MME:**

If you want to test LLaVA 1.5, you will have to clone their repo from [LLaVA](https://github.com/haotian-liu/LLaVA) and

```bash
bash examples/models/llava_next.sh
```

**Tensor Parallel Evaluation (for larger models):**

```bash
bash examples/models/tensor_parallel.sh
```

**SGLang Evaluation (for larger models):**

```bash
bash examples/models/sglang.sh
```

**vLLM Evaluation (for larger models):**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Additional Parameters:**

```bash
python3 -m lmms_eval --help
```

---

## Environment Variables

Set the following environment variables before running evaluations:

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

## Common Environment Issues

Resolve common issues by:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26; # If using numpy==2.x
python3 -m pip install sentencepiece; # May be needed for tokenizers
```

---

## Contributing

We encourage you to provide feedback, suggest new features, and contribute to the project through issues and pull requests on GitHub.

## Acknowledgements

This project is forked from [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).  For additional information, refer to their documentation [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs).

---

## Citation

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
```