<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# LMMs-Eval: Comprehensive Evaluation Suite for Large Multimodal Models

**Unleash the power of `lmms-eval` to rigorously evaluate and advance the development of cutting-edge Large Multimodal Models (LMMs).**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

➡️ **[Explore the LMMs-Eval Repository](https://github.com/EvolvingLMMs-Lab/lmms-eval)**

<br>

**Key Features:**

*   **Extensive Task Support:** Evaluate LMMs across a wide range of modalities, including text, images, video, and audio, with support for 100+ tasks.
*   **Broad Model Compatibility:** Compatible with 30+ popular LMMs, ensuring flexibility in your evaluations.
*   **Accelerated Evaluation:** Integrate vLLM for faster and more efficient model evaluation. Supports evaluation for any API-based model that follows the OpenAI API format.
*   **Reproducibility Focus:** Focused on providing clear instructions and resources for replicating results, including model details and environment setup.
*   **Regular Updates:** Stay current with the latest advancements with frequent updates, including new tasks, models, and features.
*   **Comprehensive Documentation:** Access detailed documentation for ease of use and customization.
*   **Community Support:** Engage with the LMMs-Eval community through Discord to ask questions, provide feedback, and contribute to the project.

---

## Recent Updates & Announcements

*   **[2025-07]** 🚀🚀 Released `lmms-eval-0.4` with major updates and improvements.
*   **[2025-04]** 🚀🚀 Added support for Aero-1-Audio, with batched evaluations.
*   **[2025-07]** 🎉🎉 Added support for the PhyX benchmark.
*   **[2025-06]** 🎉🎉 Added support for the VideoMathQA benchmark.
*   **[2025-02]** 🚀🚀 Integrated `vllm` and `openai_compatible` for accelerated evaluations.

<details>
<summary>See More Updates</summary>

-   [2025-01] 🎓🎓 Released new benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826).
-   [2024-12] 🎉🎉 Presented [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296).
-   [2024-11] 🔈🔊 Upgraded `lmms-eval/v0.3.0` to support audio evaluations.
-   [2024-10] 🎉🎉 Added support for NaturalBench, TemporalBench, VDC, MovieChat-1K, and Vinoground.  New Model support: AuroraCap and MovieChat.
-   [2024-09] 🎉🎉 Added support for MMSearch and MME-RealWorld.
-   [2024-09] ⚙️️⚙️️️️ Upgraded `lmms-eval` to `0.2.3` with more tasks and features.
-   [2024-08] 🎉🎉 Added support for LLaVA-OneVision, Mantis, MVBench, LongVideoBench, and MMStar.
-   [2024-07] 👨‍💻👨‍💻 Upgraded `lmms-eval/v0.2.1` with support for more models and evaluation tasks.
-   [2024-07] 🎉🎉 Released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)!
-   [2024-06] 🎬🎬 Upgraded `lmms-eval/v0.2.0` to support video evaluations.
-   [2024-03] 📝📝 Released the first version of `lmms-eval`.

</details>

---

## Why Choose LMMs-Eval?

In the rapidly evolving landscape of Large Multimodal Models, LMMs-Eval offers a comprehensive and streamlined solution for evaluating and comparing model performance.  Built upon the design of `lm-evaluation-harness`, LMMs-Eval provides a robust framework for evaluating a wide range of LMMs, enabling researchers and developers to stay at the forefront of multimodal AI.

---

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

For development:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv venv dev
source dev/bin/activate
uv pip install -e .
```

<details>
<summary>Reproducing LLaVA-1.5 Paper Results</summary>
<br>
Check the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) to reproduce LLaVA-1.5 paper results. Refer to the [results check](miscs/llava_result_check.md) for dealing with small variations.
</details>

If you need to test on caption datasets like `coco`, `refcoco`, and `nocaps`, you will need `java==1.8.0`.

```bash
conda install openjdk=8
```

If you want to test [VILA](https://github.com/NVlabs/VILA), install the following dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

## Detailed Evaluation Results
<details>
<summary>LMMs-Eval detailed evaluation results table</summary>
<br>
We provide a Google Sheet for the detailed results of the LLaVA series models on different datasets. You can access the sheet [here](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing).
</details>

## Usage Examples

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
*   **Tensor Parallel for Larger Models (llava-next-72b):**
    ```bash
    bash examples/models/tensor_parallel.sh
    ```
*   **SGLang for Larger Models (llava-next-72b):**
    ```bash
    bash examples/models/sglang.sh
    ```
*   **vLLM for Larger Models (llava-next-72b):**
    ```bash
    bash examples/models/vllm_qwen2vl.sh
    ```

*   **More Parameters:**

    ```bash
    python3 -m lmms_eval --help
    ```

*   **Environment Variables:**
    ```bash
    export OPENAI_API_KEY="<YOUR_API_KEY>"
    export HF_HOME="<Path to HF cache>"
    export HF_TOKEN="<YOUR_API_KEY>"
    export HF_HUB_ENABLE_HF_TRANSFER="1"
    export REKA_API_KEY="<YOUR_API_KEY>"
    ```

## Common Environment Issues
```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# If you are using numpy==2.x, sometimes may causing errors
python3 -m pip install numpy==1.26;
# Someties sentencepiece are required for tokenizer to work
python3 -m pip install sentencepiece;
```

## Adding Custom Models and Datasets

Please refer to our [documentation](docs/README.md) for details.

---

## Acknowledgements

This project is inspired by and builds upon the work of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).  We recommend reviewing the [lm-evaluation-harness documentation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for related information.

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