<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# LMMs-Eval: Your Comprehensive Toolkit for Evaluating Large Multimodal Models

**Evaluate and benchmark your LMMs with ease!** [Explore the original repository](https://github.com/EvolvingLMMs-Lab/lmms-eval)

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> `LMMs-Eval` provides a robust and efficient framework for evaluating and comparing Large Multimodal Models (LMMs), supporting a wide range of tasks across text, image, video, and audio modalities.

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features

*   **Comprehensive Task Coverage:** Evaluate your LMMs on a vast array of tasks including text, image, video, and audio.
*   **Broad Model Support:**  Compatible with over 30 different LMMs, including popular architectures.
*   **Efficient Evaluation:** Optimized for speed and accuracy, ensuring reliable benchmarking results.
*   **Reproducibility:** Installation using `uv` ensures consistent environments and reproducible results.
*   **Extensive Documentation:** Clear documentation and examples to guide you through the evaluation process.
*   **Active Development:** Benefit from frequent updates, new features, and community contributions.

## What's New
*   **[July 2025]** `lmms-eval-0.4` major update with new features and improvements. [Release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md)
*   **[July 2025]** New task [PhyX](https://phyx-bench.github.io/) for physics-grounded reasoning.
*   **[July 2025]** New task [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA) for mathematical reasoning in educational videos.
*   **[April 2025]** Evaluation support for [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/).
*   **[Feb 2025]** Integration of [`vllm`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/544) for accelerated evaluation and [`openai_compatible`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/546) to support any API-based model that follows the OpenAI API format.

## Installation

### Using uv (Recommended)

This uses the same package versions for all users to guarantee identical test environments.

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

Refer to the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) to reproduce LLaVA-1.5 results.  See [results check](miscs/llava_result_check.md) for results across different environments.

</details>

If you want to test on caption datasets (`coco`, `refcoco`, `nocaps`), you will need `java==1.8.0`:
```bash
conda install openjdk=8
```
Check your java version: `java -version`

<details>
<summary>LMMs Evaluation Results</summary>

Google Sheets for detailed results: [LMMs Evaluation Results](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing)
Raw data exported from Weights & Biases: [LMMs raw data](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing)

</details>

If you want to test [VILA](https://github.com/NVlabs/VILA), install:
```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

## Getting Started

### Usages

> More examples can be found in [examples/models](examples/models)

**Evaluation of OpenAI-Compatible Model**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Evaluation of vLLM**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Evaluation of LLaVA-OneVision**

```bash
bash examples/models/llava_onevision.sh
```

**Evaluation of LLaMA-3.2-Vision**

```bash
bash examples/models/llama_vision.sh
```

**Evaluation of Qwen2-VL**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**Evaluation of LLaVA on MME**

```bash
bash examples/models/llava_next.sh
```

**Evaluation with tensor parallel for bigger model (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**Evaluation with SGLang for bigger model (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**Evaluation with vLLM for bigger model (llava-next-72b)**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**More Parameters**

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

## Contributing

We encourage contributions! Please provide feedback on desired features and how to improve the library, or ask questions via issues or pull requests on GitHub.

## Add Customized Model and Dataset

Please refer to our [documentation](docs/README.md).

## Acknowledgements

`lmms_eval` is based on [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

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