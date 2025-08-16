<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png" alt="LMMs-Eval Logo">
</p>

# LMMs-Eval: The Premier Evaluation Suite for Large Multimodal Models

**Tired of scattered benchmarks and inconsistent evaluations for your multimodal models?** LMMs-Eval provides a comprehensive and efficient solution for assessing the performance of Large Multimodal Models (LMMs) across various tasks, offering a unified platform for research and development. ([Original Repository](https://github.com/EvolvingLMMs-Lab/lmms-eval))

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features of LMMs-Eval:

*   **Extensive Task Coverage:** Evaluate LMMs across a broad spectrum of text, image, video, and audio tasks.
*   **Wide Model Support:**  Compatible with over 30 different LMM architectures, constantly updated with the latest models.
*   **Efficient Evaluation:**  Optimized framework for fast and consistent evaluation, saving you valuable time.
*   **Reproducibility:** Offers detailed instructions and scripts to reproduce key results, promoting reliable research.
*   **OpenAI-Compatible Model Support:** Evaluate any API-based models that follow the OpenAI API format.
*   **VLLM Integration:** Benefit from accelerated evaluations with VLLM integration.
*   **Comprehensive Documentation and Support:**  Clear documentation and an active community ensure ease of use and support.

---
## Updates and Announcements
Stay informed with the latest developments, including new benchmarks, model integrations, and feature releases.

-   **[2025-07]**: Release of `lmms-eval-0.4`, a major update with new features and improvements.
-   **[2025-04]**: Introduction of Aero-1-Audio — a compact yet mighty audio model with batched evaluations support.
-   **[2025-07]**: Integration of new task [PhyX] (https://phyx-bench.github.io/) to test models for physics-grounded reasoning in visual scenarios.
-   **[2025-06]**: Integration of new task [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA) to evaluate mathematical reasoning in educational videos.
-   **[2025-02]**: Integration of `vllm` and `openai_compatible` features for faster evaluation and OpenAI API format support.
-   **[2025-01]**: Release of new benchmark: [Video-MMMU](https://arxiv.org/abs/2501.13826)

<details>
<summary>See all updates</summary>

-   **[2025-01]**: Released the new benchmark: [Video-MMMU](https://arxiv.org/abs/2501.13826).
-   **[2024-12]**: Presented [MME-Survey](https://arxiv.org/pdf/2411.15296), jointly with [MME Team](https://github.com/BradyFU/Video-MME) and [OpenCompass Team](https://github.com/open-compass).
-   **[2024-11]**: Support for audio evaluations for audio models like Qwen2-Audio and Gemini-Audio across tasks such as AIR-Bench, Clotho-AQA, LibriSpeech, and more.
-   **[2024-10]**: Integration of new task [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench).
-   **[2024-10]**: Integration of new task [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench).
-   **[2024-10]**: Integration of new tasks [VDC](https://rese1f.github.io/aurora-web/), [MovieChat-1K](https://rese1f.github.io/MovieChat/), and [Vinoground](https://vinoground.github.io/), along with new models: [AuroraCap](https://github.com/rese1f/aurora) and [MovieChat](https://github.com/rese1f/MovieChat).
-   **[2024-09]**: Integration of new tasks [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/).
-   **[2024-09]**: Upgrade `lmms-eval` to `0.2.3` with more tasks and features.
-   **[2024-08]**: Integration of new model [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), and new tasks [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158).
-   **[2024-07]**: Upgrade to `lmms-eval/v0.2.1` to support more models, including [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA), [InternVL-2](https://github.com/OpenGVLab/InternVL), [VILA](https://github.com/NVlabs/VILA), and many more evaluation tasks.
-   **[2024-07]**: Released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench).
-   **[2024-06]**: Upgrade to `lmms-eval/v0.2.0` to support video evaluations for video models like LLaVA-NeXT Video and Gemini 1.5 Pro.
-   **[2024-03]**: Released the first version of `lmms-eval`
</details>

---

## Installation

### Using `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

### For Development
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv venv dev
source dev/bin/activate
uv pip install -e .
```

<details>
<summary>Reproducing LLaVA-1.5 Results</summary>

Check the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) to reproduce LLaVA-1.5's paper results. We found torch/cuda versions difference would cause small variations in the results, we provide the [results check](miscs/llava_result_check.md) with different environments.

</details>

**Dependencies**

Install java 1.8 to support `pycocoeval` API for testing on `coco`, `refcoco`, and `nocaps` datasets using:

```bash
conda install openjdk=8
```
Confirm using `java -version`.

<details>
<summary>LMMs-Eval Results</summary>

We provide a Google Sheet for the detailed results of the LLaVA series models on different datasets. Access the sheet [here](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing).
</details>

If you want to test [VILA](https://github.com/NVlabs/VILA), you should install the following dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

---

## Usage Examples

> More examples can be found in [examples/models](examples/models)

**Evaluate OpenAI-Compatible Models**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Evaluate with vLLM**

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

Set the following environment variables before running experiments:
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

If you encounter issues with httpx or protobuf, try:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# If you are using numpy==2.x, sometimes may causing errors
python3 -m pip install numpy==1.26;
# Someties sentencepiece are required for tokenizer to work
python3 -m pip install sentencepiece;
```

---

## Contributing

We welcome feedback and contributions! Please submit issues or pull requests on [GitHub](https://github.com/EvolvingLMMs-Lab/lmms-eval) to help improve the library.

## Add Customized Model and Dataset
Please refer to our [documentation](docs/README.md).

## Acknowledgements

LMMs-Eval is a fork of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). We recommend you to read through the [docs of lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for relevant information.

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