<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png" alt="lmms-eval Logo">
</p>

# lmms-eval: The Comprehensive Evaluation Suite for Large Multimodal Models

**Accelerate your LMM research and development with `lmms-eval`, your go-to toolkit for evaluating large multimodal models across various tasks!** ([Original Repository](https://github.com/EvolvingLMMs-Lab/lmms-eval))

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features

*   **Comprehensive Task Support:** Evaluate LMMs across a wide range of text, image, video, and audio tasks.
*   **Extensive Model Compatibility:**  Supports a diverse set of over 30 models, including the latest advancements.
*   **Accelerated Evaluation:** Integrated with vLLM and OpenAI API compatibility for efficient model evaluation.
*   **Reproducibility:**  Offers resources for reproducing results, including environment scripts and results checks.
*   **Continuous Updates:** Regularly updated with new benchmarks, models, and features.
*   **Active Community:** Engage with the community via Discord and GitHub for support and contributions.

## Announcements

*   **[2025-07]**: Released `lmms-eval-0.4` with major updates and improvements. Refer to the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md) for more details, users using `lmms-eval-0.3` can refer to the branch `stable/v0d3`.
*   **[2025-07]**: Support for [PhyX](https://phyx-bench.github.io/), a physics-grounded reasoning benchmark.
*   **[2025-06]**: Support for [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA), for mathematical reasoning in educational videos.
*   **[2025-04]**:  Introduced Aero-1-Audio with batched evaluation support.
*   **[2025-02]**: Integrated vLLM and OpenAI API support for accelerated and broader model evaluation.
*   **(See details below for more recent additions)**

<details>
<summary>Recent Updates and Features</summary>

-   [2025-01] 🎓🎓  New benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826).
-   [2024-12] 🎉🎉 Presented [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296).
-   [2024-11] 🔈🔊  Added audio evaluation support for audio models like Qwen2-Audio and Gemini-Audio in `lmms-eval/v0.3.0`.
-   [2024-10] 🎉🎉  Support for [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench)
-   [2024-10] 🎉🎉  Support for [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench).
-   [2024-10] 🎉🎉  Support for [VDC](https://rese1f.github.io/aurora-web/), [MovieChat-1K](https://rese1f.github.io/MovieChat/), [Vinoground](https://vinoground.github.io/). New models: [AuroraCap](https://github.com/rese1f/aurora) and [MovieChat](https://rese1f/MovieChat).
-   [2024-09] 🎉🎉  Support for [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/).
-   [2024-09] ⚙️️⚙️️️️  Upgraded `lmms-eval` to `0.2.3`.
-   [2024-08] 🎉🎉  Support for [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), new tasks [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158).
-   [2024-07] 👨‍💻👨‍💻  Upgraded `lmms-eval/v0.2.1` to support more models and tasks.
-   [2024-07] 🎉🎉  Released [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)!
-   [2024-06] 🎬🎬  Upgraded `lmms-eval/v0.2.0` to support video evaluations.
-   [2024-03] 📝📝  Released the first version of `lmms-eval`.

</details>

## Installation

**Prerequisites**: Python 3.12, `java==1.8.0` (for pycocoeval on caption datasets).

**Installation (using uv):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

**For Development:**

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv venv dev
source dev/bin/activate
uv pip install -e .
```

**Java Setup**
If you want to test on caption dataset such as `coco`, `refcoco`, and `nocaps`, you will need to have `java==1.8.0` to let pycocoeval api to work. If you don't have it, you can install by using conda
```
conda install openjdk=8
```
you can then check your java version by `java -version` 

**Reproducing LLaVA-1.5 Results:**  Refer to `miscs/repr_scripts.sh` and `miscs/repr_torch_envs.txt` for reproducing LLaVA-1.5's paper results.  Check `miscs/llava_result_check.md` for results validation in different environments.

<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>
<br>

As demonstrated by the extensive table below, we aim to provide detailed information for readers to understand the datasets included in lmms-eval and some specific details about these datasets (we remain grateful for any corrections readers may have during our evaluation process).

We provide a Google Sheet for the detailed results of the LLaVA series models on different datasets. You can access the sheet [here](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing). It's a live sheet, and we are updating it with new results.

<p align="center" width="100%">
<img src="https://i.postimg.cc/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%">
</p>

We also provide the raw data exported from Weights & Biases for the detailed results of the LLaVA series models on different datasets. You can access the raw data [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

</details>
<br>

If you want to test [VILA](https://github.com/NVlabs/VILA), you should install the following dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

## Usage Examples

> More examples can be found in [examples/models](examples/models)

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

*   **LLaVA on MME Evaluation:**
    ```bash
    bash examples/models/llava_next.sh
    ```

*   **Tensor Parallel Evaluation (for larger models):**
    ```bash
    bash examples/models/tensor_parallel.sh
    ```

*   **SGLang Evaluation (for larger models):**
    ```bash
    bash examples/models/sglang.sh
    ```

*   **vLLM Evaluation (for larger models):**
    ```bash
    bash examples/models/vllm_qwen2vl.sh
    ```

*   **For More Parameters:**  Run `python3 -m lmms_eval --help`

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

## Troubleshooting

**Common issues:**

*   **httpx/protobuf errors:**
    ```bash
    python3 -m pip install httpx==0.23.3;
    python3 -m pip install protobuf==3.20;
    python3 -m pip install numpy==1.26;
    python3 -m pip install sentencepiece;
    ```

## Contributing

We welcome contributions!  Please submit feature requests, bug reports, and pull requests on GitHub.

## Acknowledgements

`lmms-eval` is based on the design of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

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