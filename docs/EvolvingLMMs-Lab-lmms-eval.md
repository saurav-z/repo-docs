<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png" alt="LMMs-Eval Logo">
</p>

# LMMs-Eval: The Premier Evaluation Suite for Large Multimodal Models

**Accelerate LMM development with `lmms-eval`, your go-to open-source evaluation framework.** ([Back to Repository](https://github.com/EvolvingLMMs-Lab/lmms-eval))

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features:

*   **Comprehensive Support:** Evaluate LMMs across text, image, video, and audio tasks.
*   **Extensive Task Coverage:**  Includes over 100 supported tasks spanning various multimodal benchmarks.
*   **Broad Model Compatibility:** Supports 30+ models, with integrations for popular architectures.
*   **Accelerated Evaluation:**  Leverages vLLM and OpenAI API compatibility for efficient model assessment.
*   **Active Development:**  Regularly updated with new tasks, models, and features.
*   **Reproducibility:**  Provides scripts and resources to reproduce results, fostering reliability.
*   **Community Focused:**  Encourages contributions and feedback through GitHub issues and discussions.

---

## What's New

-   **[2025-07]**  🚀🚀 Released `lmms-eval-0.4` with major updates. See [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md) for details.
-   **[2025-07]** 🎉🎉  Added support for the new [PhyX](https://phyx-bench.github.io/) benchmark for physics reasoning.
-   **[2025-06]** 🎉🎉  Added support for the new [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA) benchmark for mathematical reasoning in videos.
-   **[2025-04]** 🚀🚀 Introduced Aero-1-Audio — now supports batched evaluations.
-   **[2025-02]** 🚀🚀 Integrated `vllm` for faster evaluations. Added `openai_compatible` support.

<details>
<summary>See More Updates</summary>

-   [2025-01] 🎓🎓  Released new benchmark: [Video-MMMU](https://arxiv.org/abs/2501.13826)
-   [2024-12]  Presented [MME-Survey](https://arxiv.org/pdf/2411.15296)
-   [2024-11] 🔈🔊  Audio evaluations for audio models like Qwen2-Audio and Gemini-Audio
-   [2024-10] 🎉🎉  Added support for [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench), and more.
-   [2024-09] 🎉🎉  Added support for [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/).
-   [2024-09] ⚙️️⚙️️️️ Upgraded to `0.2.3` with more tasks and features.
-   [2024-08] 🎉🎉 Added support for [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), new tasks [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158).
-   [2024-07] 👨‍💻👨‍💻  Upgraded to `0.2.1` with support for [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA), [InternVL-2](https://github.com/OpenGVLab/InternVL), [VILA](https://github.com/NVlabs/VILA), and more.
-   [2024-07] 🎉🎉 Released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)!
-   [2024-06] 🎬🎬  Upgraded to `0.2.0` to support video evaluations.
-   [2024-03] 📝📝 Released the first version of `lmms-eval`.

</details>

## Installation

Install `lmms-eval` using the following methods:

**Direct Installation (Recommended)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

**Development Installation**

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv venv dev
source dev/bin/activate
uv pip install -e .
```

<details>
<summary>Reproducing LLaVA-1.5 Results</summary>

Follow the instructions in [miscs/repr_scripts.sh](miscs/repr_scripts.sh) and consult [miscs/repr_torch_envs.txt](miscs/repr_torch_envs.txt) to reproduce LLaVA-1.5 results.  Check the [results check](miscs/llava_result_check.md) for environment-specific result variations.

</details>

**Java Dependency (for coco, refcoco, nocaps)**

```bash
conda install openjdk=8
```
Then check your java version with `java -version`.

<details>
<summary>Evaluation Results for LLaVA Models</summary>

[Detailed results](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing) and [raw data](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing) are available.

<p align="center" width="100%">
<img src="https://i.postimg.cc/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%">
</p>
</details>

**VILA Dependencies:**

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

## Usage Examples

>  Find more examples in the [examples/models](examples/models) directory.

*   **OpenAI-Compatible Model Evaluation:**
    ```bash
    bash examples/models/openai_compatible.sh
    bash examples/models/xai_grok.sh
    ```

*   **vLLM Evaluation:**
    ```bash
    bash examples/models/vllm_qwen2vl.sh
    ```

*   **LLaVA-OneVision Evaluation:**
    ```bash
    bash examples/models/llava_onevision.sh
    ```

*   **Llama-3-Vision Evaluation:**
    ```bash
    bash examples/models/llama_vision.sh
    ```

*   **Qwen2-VL Evaluation:**
    ```bash
    bash examples/models/qwen2_vl.sh
    bash examples/models/qwen2_5_vl.sh
    ```

*   **LLaVA on MME:**
    ```bash
    bash examples/models/llava_next.sh
    ```

*   **Tensor Parallel Evaluation (llava-next-72b):**
    ```bash
    bash examples/models/tensor_parallel.sh
    ```

*   **SGLang Evaluation (llava-next-72b):**
    ```bash
    bash examples/models/sglang.sh
    ```

*   **vLLM Evaluation (llava-next-72b):**
    ```bash
    bash examples/models/vllm_qwen2vl.sh
    ```

**Command Line Help:**

```bash
python3 -m lmms_eval --help
```

## Environment Variables

Set these environment variables before running evaluations:

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include: ANTHROPIC_API_KEY, DASHSCOPE_API_KEY, etc.
```

## Common Issues and Solutions

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26;
python3 -m pip install sentencepiece;
```

## Customization

Refer to the [documentation](docs/README.md) for information on adding custom models and datasets.

## Acknowledgements

`lmms-eval` is a fork of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).  See the [lm-eval-harness docs](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for more information.

---

## Key Modifications from the Original API

*   Context building now focuses on the image and document processing during model response.
*   `Instance.args` (lmms_eval/api/instance.py) contains a list of images.
*   Model support is currently implemented per-model due to HF format differences.  (Ongoing effort to unify).

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