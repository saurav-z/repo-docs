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

**FastDeploy** is a powerful toolkit designed to accelerate the deployment of Large Language Models (LLMs) and Visual Language Models (VLMs), offering production-ready solutions based on PaddlePaddle.  [Explore the original repository](https://github.com/PaddlePaddle/FastDeploy).

## Key Features:

*   **Optimized Performance:** Achieve high throughput and low latency with cutting-edge acceleration techniques.
    *   **Load-Balanced PD Disaggregation:**  Industrial-grade solution featuring context caching and dynamic instance role switching. Optimizes resource utilization while balancing SLO compliance and throughput.
    *   **Unified KV Cache Transmission:** Lightweight high-performance transport library with intelligent NVLink/RDMA selection.
    *   **Advanced Acceleration:** Leverage speculative decoding, Multi-Token Prediction (MTP), and Chunked Prefill for faster inference.
*   **Wide Hardware Support:** Deploy on various hardware platforms.
    *   NVIDIA GPUs
    *   Kunlunxin XPUs
    *   Hygon DCUs
    *   Ascend NPUs (coming soon)
    *   Iluvatar GPUs
    *   Enflame GCUs
    *   MetaX GPU
*   **Flexible Deployment Options:** Easily integrate FastDeploy into your existing infrastructure.
    *   **OpenAI API Server and vLLM Compatible:** One-command deployment with [vLLM](https://github.com/vllm-project/vllm/) interface compatibility.
    *   **Offline and Online Serving:**  Support for both offline inference and online service deployment scenarios.
*   **Comprehensive Quantization:** Optimize model size and performance.
    *   Supports various quantization formats: W8A16, W8A8, W4A16, W4A8, W2A16, FP8, and more.
*   **Ease of Use:** Get started quickly with comprehensive documentation and examples.

## What's New

*   **[2025-09] 🔥 FastDeploy v2.2:** Compatibility with HuggingFace models, performance optimizations, and support for [baidu/ERNIE-21B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking)!
*   **[2025-08] 🔥 FastDeploy v2.1:** Introduced a new KV Cache scheduling strategy, expanded support for PD disaggregation and CUDA Graph, and enhanced hardware support.
*   **[2025-07]** The FastDeploy 2.0 Inference Deployment Challenge is live!
*   **[2025-06] 🔥 FastDeploy v2.0:** Supports inference and deployment for ERNIE 4.5.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.10 ~ 3.12

## Installation

Install FastDeploy for your specific hardware:

*   [NVIDIA GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/)
*   [Kunlunxin XPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/kunlunxin_xpu/)
*   [Iluvatar GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/iluvatar_gpu/)
*   [Enflame GCU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/Enflame_gcu.md)
*   [Hygon DCU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/hygon_dcu.md)
*   [MetaX GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/metax_gpu.md)

*Note: Support for Ascend NPU is under development.*

## Get Started

Explore the following resources to quickly learn how to use FastDeploy:

*   [10-Minutes Quick Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/quick_start.md)
*   [ERNIE-4.5 Large Language Model Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/ernie-4.5.md)
*   [ERNIE-4.5-VL Multimodal Model Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/ernie-4.5-vl.md)
*   [Offline Inference Development](https://paddlepaddle.github.io/FastDeploy/docs/offline_inference.md)
*   [Online Service Deployment](https://paddlepaddle.github.io/FastDeploy/docs/online_serving/README.md)
*   [Best Practices](https://paddlepaddle.github.io/FastDeploy/docs/best_practices/README.md)

## Supported Models

*   [Full Supported Models List](https://paddlepaddle.github.io/FastDeploy/docs/supported_models.md)

## Advanced Usage

*   [Quantization](https://paddlepaddle.github.io/FastDeploy/docs/quantization/README.md)
*   [PD Disaggregation Deployment](https://paddlepaddle.github.io/FastDeploy/docs/features/disaggregated.md)
*   [Speculative Decoding](https://paddlepaddle.github.io/FastDeploy/docs/features/speculative_decoding.md)
*   [Prefix Caching](https://paddlepaddle.github.io/FastDeploy/docs/features/prefix_caching.md)
*   [Chunked Prefill](https://paddlepaddle.github.io/FastDeploy/docs/features/chunked_prefill.md)

## Acknowledgement

FastDeploy is licensed under the [Apache-2.0 open-source license](./LICENSE).  We are grateful for the contributions of [vLLM](https://github.com/vllm-project/vllm) code, which was referenced and incorporated to maintain interface compatibility.