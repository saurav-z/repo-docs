<p align="center" width="70%">
<img src="https://i.postimg.com/KvkLzbF9/WX20241212-014400-2x.png" alt="LMMs-Eval Logo">
</p>

# LMMs-Eval: Your Comprehensive Toolkit for Evaluating Large Multimodal Models (LMMs)

**Accelerate your LMM research with LMMs-Eval, the go-to evaluation suite for text, image, video, and audio tasks.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

**[Explore the LMMs-Eval Repository](https://github.com/EvolvingLMMs-Lab/lmms-eval)**

---

## Key Features

*   **Extensive Task Support:** Evaluate LMMs across a wide range of modalities, including text, image, video, and audio, covering **100+ tasks**.
*   **Broad Model Compatibility:** Supports **30+ models**, including popular LMMs like LLaVA, Qwen-VL, and Gemini.
*   **Accelerated Evaluation:** Integrates technologies like vLLM and OpenAI-compatible APIs for faster evaluation.
*   **Reproducibility:**  Provides scripts and documentation to reproduce results.
*   **Customization:** Easily add your own models and datasets.
*   **Comprehensive Documentation:** Detailed documentation and examples to guide you through the evaluation process.

---

## What's New

*   **Recent Updates:** Stay informed about the latest features and improvements in our [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md), including the major `lmms-eval-0.4` release!
*   **New Task Support:**  We continuously add new benchmarks to assess your LMMs:  PhyX, VideoMathQA, NaturalBench, TemporalBench, VDC, MovieChat-1K, and Vinoground.
*   **Model Updates:** We've added evaluation support for Aero-1-Audio (with batched evaluations), LLaMA-3.2-Vision,  and more!
*   **Performance Enhancements:**  Integrated vLLM for accelerated evaluations and `openai_compatible` for easy integration of API-based models.

<details>
<summary>See Past Updates</summary>

*   **Video Evaluation Support:** Evaluate video models across various tasks, including EgoSchema, PerceptionTest, VideoMME, and more.
*   **Audio Evaluation Support:**  Support for audio evaluations for audio models like Qwen2-Audio and Gemini-Audio across tasks such as AIR-Bench, Clotho-AQA, LibriSpeech, and more.
*   **Benchmarks and Technical Report:** We have released our new benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826) and [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296) and the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)!

</details>

---

## Installation

**Prerequisites**:  Ensure you have Python 3.12 installed.  You may also need `java==1.8.0` if you are working with image captioning datasets.
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
<details>
<summary>Reproducing LLaVA-1.5 Results</summary>

Follow the instructions provided in `miscs/repr_scripts.sh` and check `miscs/repr_torch_envs.txt`.  Review the [results check](miscs/llava_result_check.md) document for further clarification.
</details>
<br>
If you want to test [VILA](https://github.com/NVlabs/VILA), you should install the following dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

---

## Usage Examples

> Refer to [examples/models](examples/models) for more detailed examples.

**Evaluate OpenAI-Compatible Models:**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Evaluate with vLLM:**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Example Evaluations:**

*   LLaVA-OneVision:  `bash examples/models/llava_onevision.sh`
*   LLaMA-3.2-Vision:  `bash examples/models/llama_vision.sh`
*   Qwen2-VL: `bash examples/models/qwen2_vl.sh`
*   More examples and parameters with `python3 -m lmms_eval --help`.

---

## Important Notes and Variables

*   **Environmental Variables:**  Set the following environment variables for optimal performance:  `OPENAI_API_KEY`, `HF_HOME`, `HF_TOKEN`, `HF_HUB_ENABLE_HF_TRANSFER`, `REKA_API_KEY`.
*   **Common Issues:**  If you encounter issues related to `httpx`, `protobuf`, or `numpy`, try the following:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26;
python3 -m pip install sentencepiece;
```

*   **Customization:** Customize your evaluation by referring to the [documentation](docs/README.md).

---

## Acknowledgements

This project is based on the foundation of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

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