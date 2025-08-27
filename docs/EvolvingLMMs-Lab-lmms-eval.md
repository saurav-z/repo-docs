<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png" alt="lmms-eval Logo">
</p>

# LMMs-Eval: The Comprehensive Evaluation Suite for Large Multimodal Models

**Accelerate your LMM research with LMMs-Eval, a powerful and versatile evaluation suite designed for large multimodal models.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

[**Visit the Original Repo on GitHub**](https://github.com/EvolvingLMMs-Lab/lmms-eval)

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features

*   **Extensive Task Support:** Evaluate LMMs across a wide range of text, image, video, and audio tasks.
*   **Model Compatibility:** Supports a diverse collection of over 30 LMMs.
*   **Modular Design:** Built upon the robust foundation of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), ensuring efficient and reliable evaluation.
*   **Reproducibility:** Provides installation and result checking scripts for reproducing paper results.
*   **Continuous Updates:**  Regularly updated with new tasks, models, and features.
*   **OpenAI API Compatibility:**  Supports evaluation of any API-based model that follows the OpenAI API format.
*   **Integration with vLLM and SGLang:**  Offers accelerated evaluation through vLLM and SGLang.

---

## Recent Updates and Announcements

*   **[2025-07]** Released `lmms-eval-0.4` with major updates and improvements. See [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md). Discussion on model eval results in [discussion](https://github.com/EvolvingLMMs-Lab/lmms-eval/discussions/779).
*   **[2025-07]**  Added support for the [PhyX](https://phyx-bench.github.io/) benchmark.
*   **[2025-06]**  Added support for the [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA) benchmark.
*   **[2025-04]**  Introduced and supported evaluation for [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/).
*   **[2025-02]** Integrated [`vllm`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/544) and  [`openai_compatible`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/546).

<details>
<summary>More Updates</summary>

*   **[2025-01]**  Released new benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826).
*   **[2024-12]** Presented [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296).
*   **[2024-11]**  Upgraded `lmms-eval/v0.3.0` to support audio evaluations.
*   **[2024-10]** Added support for [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench), [VDC](https://rese1f.github.io/aurora-web/), [MovieChat-1K](https://rese1f.github.io/MovieChat/), and [Vinoground](https://vinoground.github.io/).  Added support for models:  [AuroraCap](https://github.com/rese1f/aurora) and [MovieChat](https://rese1f/MovieChat).
*   **[2024-09]**  Added support for [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/) for inference acceleration. Upgraded `lmms-eval` to `0.2.3`.
*   **[2024-08]** Added support for [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), and new tasks.
*   **[2024-07]** The `lmms-eval/v0.2.1` was upgraded to support more models and evaluation tasks. Released [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench).
*   **[2024-06]** The `lmms-eval/v0.2.0` was upgraded to support video evaluations.
*   **[2024-03]** Released the first version of `lmms-eval`.
</details>

---

## Installation

### Using uv (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv sync
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

See [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) to reproduce LLaVA-1.5 paper results.  Check [results check](miscs/llava_result_check.md) for variations.
</details>

**Install Java (if needed for certain datasets):**

```bash
conda install openjdk=8
```

<details>
<summary>Comprehensive Evaluation Results</summary>
<br>

[LMMs-Eval Detailed Results](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing)

[Raw Data from Weights & Biases](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing)

</details>

**Install VILA dependencies:**

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

---

## Usage Examples

See [examples/models](examples/models) for detailed usage.

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

**Evaluate LLaMA-3.2-Vision:**

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

**More Parameters:**

```bash
python3 -m lmms_eval --help
```

---

## Environmental Variables

Set the following environment variables *before* running experiments:

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

**Common Issues and Solutions:**

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26;
python3 -m pip install sentencepiece;
```

---

## Adding Customized Models and Datasets

Refer to our [documentation](docs/README.md).

---

## Acknowledgements

This project is a fork of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

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