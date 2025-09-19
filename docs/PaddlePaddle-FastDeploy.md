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

--------------------------------------------------------------------------------

# FastDeploy: Deploy LLMs and VLMs with Speed and Efficiency

FastDeploy is a powerful toolkit designed to streamline the deployment of Large Language Models (LLMs) and Visual Language Models (VLMs), offering production-ready solutions optimized for performance and efficiency.  **[View the original repository on GitHub](https://github.com/PaddlePaddle/FastDeploy)**

## Key Features

*   **Optimized Inference:**  Leverages advanced techniques to accelerate LLM and VLM inference.
*   **Load-Balanced PD Disaggregation:** Industrial-grade solution featuring context caching and dynamic instance role switching. Optimizes resource utilization while balancing SLO compliance and throughput.
*   **Unified KV Cache Transmission:** Lightweight high-performance transport library with intelligent NVLink/RDMA selection.
*   **OpenAI API Server and vLLM Compatible:** One-command deployment with [vLLM](https://github.com/vllm-project/vllm/) interface compatibility.
*   **Comprehensive Quantization:** Supports various quantization formats (W8A16, W8A8, W4A16, W4A8, W2A16, FP8, and more).
*   **Advanced Acceleration:**  Includes speculative decoding, Multi-Token Prediction (MTP), and Chunked Prefill.
*   **Multi-Hardware Support:** Optimized for NVIDIA GPU, Kunlunxin XPU, Hygon DCU, Ascend NPU, Iluvatar GPU, Enflame GCU, MetaX GPU, and more.

## What's New

*   **[2025-09] 🔥 FastDeploy v2.2:**  Now offers compatibility with models in the HuggingFace ecosystem and includes support for baidu/ERNIE-21B-A3B-Thinking!
*   **[2025-08] 🔥 FastDeploy v2.1:** Introduction of a new KV Cache scheduling strategy, expanded support for PD separation and CUDA Graph, and enhanced hardware support, with performance optimizations for both service and inference engines.
*   **[2025-07] FastDeploy 2.0 Inference Deployment Challenge:** Participate in the ERNIE 4.5 series inference deployment challenge and win prizes! [Sign up here](https://www.wjx.top/vm/meSsp3L.aspx#) and see [event details](https://github.com/PaddlePaddle/FastDeploy/discussions/2728).
*   **[2025-06] 🔥 FastDeploy v2.0:** Supports inference and deployment for ERNIE 4.5. Also open-sourced industrial-grade PD disaggregation with context caching and dynamic role switching.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.10 - 3.12

## Installation

FastDeploy supports inference deployment on a variety of hardware platforms:

*   [NVIDIA GPU](docs/get_started/installation/nvidia_gpu.md)
*   [Kunlunxin XPU](docs/get_started/installation/kunlunxin_xpu.md)
*   [Iluvatar GPU](docs/get_started/installation/iluvatar_gpu.md)
*   [Enflame GCU](docs/get_started/installation/Enflame_gcu.md)
*   [Hygon DCU](docs/get_started/installation/hygon_dcu.md)
*   [MetaX GPU](docs/get_started/installation/metax_gpu.md)

**Note:**  Support for additional hardware platforms, including Ascend NPU, is actively being developed.  Stay tuned for updates!

## Get Started

Explore the following resources to begin using FastDeploy:

*   [10-Minutes Quick Deployment](docs/get_started/quick_start.md)
*   [ERNIE-4.5 Large Language Model Deployment](docs/get_started/ernie-4.5.md)
*   [ERNIE-4.5-VL Multimodal Model Deployment](docs/get_started/ernie-4.5-vl.md)
*   [Offline Inference Development](docs/offline_inference.md)
*   [Online Service Deployment](docs/online_serving/README.md)
*   [Best Practices](docs/best_practices/README.md)

## Supported Models

Find a complete list of supported models and learn how to enable the torch format:

*   [Full Supported Models List](docs/supported_models.md)

## Advanced Usage

*   [Quantization](docs/quantization/README.md)
*   [PD Disaggregation Deployment](docs/features/disaggregated.md)
*   [Speculative Decoding](docs/features/speculative_decoding.md)
*   [Prefix Caching](docs/features/prefix_caching.md)
*   [Chunked Prefill](docs/features/chunked_prefill.md)

## Acknowledgements

FastDeploy is licensed under the [Apache-2.0 open-source license](./LICENSE). During development, portions of the [vLLM](https://github.com/vllm-project/vllm) code were referenced and incorporated, for which we express our gratitude.