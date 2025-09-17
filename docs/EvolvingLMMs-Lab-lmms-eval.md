<p align="center" width="70%">
<img src="https://i.postimg.com/KvkLzbF9/WX20241212-014400-2x.png" alt="LMMs-Eval Logo">
</p>

# LMMs-Eval: Evaluate and Accelerate Large Multimodal Model Development

**LMMs-Eval is your all-in-one solution for comprehensive evaluation of Large Multimodal Models (LMMs), supporting a wide range of tasks and models to accelerate the development of AGI.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features

*   **Extensive Task Support:** Evaluate LMMs across a broad spectrum of tasks, including text, image, video, and audio.
*   **Wide Model Compatibility:** Supports over 30 different LMMs, with continuous expansion.
*   **Accelerated Evaluation:**  Integrates with tools like vLLM and SGLang for faster and more efficient model evaluation.
*   **Reproducibility Focused:** Features clear instructions and scripts for replicating results, including detailed environment setups.
*   **Community Driven:**  Active development with contributions from a dedicated community.
*   **Easy Installation:** Streamlined setup using `uv` for consistent environments, as well as alternative installation methods.
*   **Flexible Customization:**  Detailed documentation to guide users on how to add new models and datasets.
*   **Comprehensive Results:** Provides detailed and up-to-date results for various LLaVA models, offering valuable insights into performance.

---

## Announcements

*   **[2025-07]** 🚀🚀 Release of `lmms-eval-0.4` with major updates and improvements.  See [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-07]** 🎉🎉  Support for the new benchmark [PhyX](https://phyx-bench.github.io/).
*   **[2025-06]** 🎉🎉 Support for the new benchmark [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA).
*   **[2025-04]** 🚀🚀  Introduced [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/) - a compact audio model with batched evaluation support.
*   **[2025-02]** 🚀🚀 Integrated `vllm` and `openai_compatible` for faster and more flexible model evaluations.

<details>
<summary>See older announcements</summary>

*   **[2025-01]** 🎓🎓 Released new benchmark: [Video-MMMU](https://arxiv.org/abs/2501.13826).
*   **[2024-12]** 🎉🎉 Presented [MME-Survey](https://arxiv.org/pdf/2411.15296) with MME and OpenCompass Teams.
*   **[2024-11]** 🔈🔊  `lmms-eval/v0.3.0` adds support for audio evaluations.
*   **[2024-10]** 🎉🎉 Added support for NaturalBench, TemporalBench, VDC, MovieChat-1K, and Vinoground benchmarks, and AuroraCap and MovieChat models.
*   **[2024-09]** 🎉🎉 Added support for MMSearch and MME-RealWorld. `lmms-eval` upgraded to `0.2.3`.
*   **[2024-08]** 🎉🎉 Added support for LLaVA-OneVision, Mantis, MVBench, LongVideoBench, and MMStar.  Added SGlang Runtime API feature for llava-onevision.
*   **[2024-07]** 👨‍💻👨‍💻 `lmms-eval/v0.2.1` support for more models and tasks.
*   **[2024-07]** 🎉🎉 Released the technical report and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench).
*   **[2024-06]** 🎬🎬  `lmms-eval/v0.2.0` enhanced video evaluations.
*   **[2024-03]** 📝📝  Initial release of `lmms-eval`.

</details>

---

## Installation

**Recommended: Using `uv` for Consistent Environments**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv sync
uv run python -m lmms_eval --help
uv add <package>
```

**Alternative Installation:**

```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

<details>
<summary>Reproducing LLaVA-1.5 Results</summary>

Follow the instructions in `miscs/repr_scripts.sh` and consult `miscs/repr_torch_envs.txt` to reproduce LLaVA-1.5 paper results. Check the `miscs/llava_result_check.md` for results based on different environments.

</details>

**Dependencies:**

Install Java 1.8 for `pycocoeval` if testing on `coco`, `refcoco`, or `nocaps`:
```bash
conda install openjdk=8
```

If you want to test [VILA](https://github.com/NVlabs/VILA), you should install the following dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

---
<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>
<br>

Access the detailed results of the LLaVA series models on different datasets in this [Google Sheet](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing). Raw data from Weights & Biases can also be found [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

</details>
<br>

> **See the [original repo](https://github.com/EvolvingLMMs-Lab/lmms-eval) for more information.**

---

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
*   **LLaMA-3.2-Vision Evaluation:**
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
*   **For additional parameters:**
    ```bash
    python3 -m lmms_eval --help
    ```

**Environment Variables:**
Set necessary environment variables before running experiments:

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other environment variables
```

**Troubleshooting Common Issues**

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26;
python3 -m pip install sentencepiece;
```

## Adding Customized Models and Datasets

Refer to the [documentation](docs/README.md) for instructions.

---
## Acknowledgements

This project is based on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

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