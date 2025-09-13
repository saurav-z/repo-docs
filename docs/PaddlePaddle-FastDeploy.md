<p align="center">
  <a href="https://github.com/PaddlePaddle/FastDeploy/releases"><img src="https://github.com/user-attachments/assets/42b0039f-39e3-4279-afda-6d1865dfbffb" width="500"></a>
</p>
<p align="center">
    <a href=""><img src="https://img.shields.io/badge/python-3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/FastDeploy?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/FastDeploy?color=3af"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/FastDeploy?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/FastDeploy?color=ccf"></a>

</p>

<p align="center">
     <a href="https://trendshift.io/repositories/4046" target="_blank"><img src="https://trendshift.io/api/badge/repositories/4046" alt="PaddlePaddle%2FFastDeploy | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a></br>
    <a href="https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/"><b> Installation </b></a>
    |
    <a href="https://paddlepaddle.github.io/FastDeploy/get_started/quick_start"><b> Quick Start </b></a>
    |
    <a href="https://paddlepaddle.github.io/FastDeploy/supported_models/"><b> Supported Models </b></a>

</p>

---

# FastDeploy: Deploy LLMs and VLMs with Lightning Speed and Efficiency

**FastDeploy** provides a production-ready toolkit for deploying Large Language Models (LLMs) and Visual Language Models (VLMs) based on PaddlePaddle, enabling rapid inference and deployment. Learn more on the [original repo](https://github.com/PaddlePaddle/FastDeploy).

## Key Features

*   **🚀 Production-Ready Deployment:** Get out-of-the-box solutions for efficient LLM and VLM deployment.
*   **🔄 Load-Balanced PD Disaggregation:** Industrial-grade solution optimizing resource utilization with context caching and dynamic instance role switching.
*   **🤝 OpenAI API & vLLM Compatibility:**  Seamless integration with vLLM and OpenAI API server for ease of use.
*   **🧮 Comprehensive Quantization Support:** Supports a wide range of quantization formats including W8A16, W8A8, W4A16, and more for model optimization.
*   **⏩ Advanced Acceleration Techniques:** Leverages speculative decoding, Multi-Token Prediction (MTP), and Chunked Prefill for faster inference.
*   **🖥️ Multi-Hardware Support:** Optimized for NVIDIA GPUs, Kunlunxin XPUs, Hygon DCUs, Ascend NPUs, and more.

## Recent Updates

*   **[2024-09] v2.2 Released!** Featuring HuggingFace ecosystem compatibility, performance improvements, and support for baidu/ERNIE-21B-A3B-Thinking.
*   **[2024-08] v2.1 Released:** Introduces a new KV Cache scheduling strategy, enhanced PD separation and CUDA Graph support, plus hardware support and optimizations.
*   **[2024-07] FastDeploy 2.0 Inference Deployment Challenge:** Participate in the challenge for ERNIE 4.5 models and win prizes! ([Sign up here](https://www.wjx.top/vm/meSsp3L.aspx#) [Event details](https://github.com/PaddlePaddle/FastDeploy/discussions/2728))
*   **[2024-06] v2.0 Released:** Support for ERNIE 4.5 inference/deployment, industrial-grade PD disaggregation, context caching, and dynamic role switching.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.10 ~ 3.12

## Installation

FastDeploy supports a variety of hardware platforms.  Install FastDeploy with these guides:

*   [NVIDIA GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu.md)
*   [Kunlunxin XPU](docs/get_started/installation/kunlunxin_xpu.md)
*   [Iluvatar GPU](docs/get_started/installation/iluvatar_gpu.md)
*   [Enflame GCU](docs/get_started/installation/Enflame_gcu.md)
*   [Hygon DCU](docs/get_started/installation/hygon_dcu.md)
*   [MetaX GPU](docs/get_started/installation/metax_gpu.md)

**Note:**  We are constantly expanding hardware support. Stay tuned for updates on Ascend NPU and other platforms.

## Get Started

Explore FastDeploy with these helpful resources:

*   [10-Minutes Quick Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/quick_start.md)
*   [ERNIE-4.5 Large Language Model Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/ernie-4.5.md)
*   [ERNIE-4.5-VL Multimodal Model Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/ernie-4.5-vl.md)
*   [Offline Inference Development](docs/offline_inference.md)
*   [Online Service Deployment](docs/online_serving/README.md)
*   [Best Practices](docs/best_practices/README.md)

## Supported Models

*   [Full Supported Models List](https://paddlepaddle.github.io/FastDeploy/supported_models/)

## Advanced Usage

*   [Quantization](docs/quantization/README.md)
*   [PD Disaggregation Deployment](docs/features/disaggregated.md)
*   [Speculative Decoding](docs/features/speculative_decoding.md)
*   [Prefix Caching](docs/features/prefix_caching.md)
*   [Chunked Prefill](docs/features/chunked_prefill.md)

## Acknowledgement

FastDeploy is licensed under the [Apache-2.0](LICENSE). We are grateful to the [vLLM](https://github.com/vllm-project/vllm) team for their contributions, as parts of their code were referenced and incorporated to maintain interface compatibility.