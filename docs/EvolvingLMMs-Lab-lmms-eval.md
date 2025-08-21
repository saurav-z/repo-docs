<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png" alt="LMMs-Eval Logo">
</p>

# LMMs-Eval: The Premier Evaluation Suite for Large Multimodal Models

**Effortlessly evaluate and compare your Large Multimodal Models (LMMs) with `lmms-eval`—your one-stop solution for comprehensive benchmarking!** ([Original Repository](https://github.com/EvolvingLMMs-Lab/lmms-eval))

## Key Features

*   **Comprehensive Task Support:** Evaluate LMMs across a vast array of text, image, video, and audio tasks.
*   **Wide Model Compatibility:**  Supports over 30 popular LMMs, with continuous expansion.
*   **Reproducibility Focus:**  Includes detailed instructions and scripts to reproduce results from leading research papers.
*   **Rapid Evaluation:** Integrates tools like vLLM and supports OpenAI API compatible models for accelerated benchmarking.
*   **Extensive Documentation:**  Detailed documentation and examples for easy integration and customization.
*   **Community-Driven:** Actively maintained with contributions from the community, ensuring the suite stays current and relevant.

## Core Capabilities

*   **Supports Diverse Modalities:** Evaluate models on text, image, video, and audio tasks.
*   **Streamlined Evaluation:** Integrates tools like vLLM for accelerated benchmarking.
*   **API Compatibility:** Includes OpenAI API compatible models.
*   **Reproducibility:** Provides instructions to reproduce key results.
*   **Ongoing Development:** The library is under active development and provides the latest tasks and models.

## What's New

*   **Latest Release (v0.4):** Major update with new features and improvements.  See [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **Aero-1-Audio Support:** Evaluate the compact yet powerful Aero-1-Audio model with batched evaluations.
*   **New Benchmark Integration:** Support for PhyX and VideoMathQA benchmarks.
*   **Enhanced Efficiency:** Integration of `vllm` and `openai_compatible` for faster and more flexible evaluations.
*   **Expanded Task Coverage:** Inclusion of recent benchmarks like Video-MMMU, NaturalBench, TemporalBench, and more.
*   **New Model Support:** Includes LLaVA-OneVision, LLaMA-3.2-Vision, and others.
*   **Video Evaluation:** Includes evaluation support for video models like LLaVA-NeXT Video and Gemini 1.5 Pro.
*   **Audio Evaluation:** Supports evaluation for audio models such as Qwen2-Audio and Gemini-Audio.

## Installation

**Using `uv`:**

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

**Note:** You might need `java==1.8.0` to run caption datasets like `coco`, `refcoco`, and `nocaps`. Install with `conda install openjdk=8`.

## Usage Examples

*   **Evaluate OpenAI-Compatible Model:**
    ```bash
    bash examples/models/openai_compatible.sh
    bash examples/models/xai_grok.sh
    ```

*   **Evaluate vLLM:**
    ```bash
    bash examples/models/vllm_qwen2vl.sh
    ```

*   **Evaluate LLaVA-OneVision:**
    ```bash
    bash examples/models/llava_onevision.sh
    ```

*   **Explore More Examples:** Navigate to the `examples/models` directory for detailed usage examples.

## Environment Variables

Set the following environment variables before running:

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include 
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

## Add Your Own Models and Datasets

Refer to the detailed [documentation](docs/README.md) for instructions on adding custom models and datasets.

## Acknowledgements

This project builds upon and extends the capabilities of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).  We encourage you to review the documentation there for valuable information.

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