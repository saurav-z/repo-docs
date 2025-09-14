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

**FastDeploy** empowers developers to rapidly deploy and optimize large language models (LLMs) and visual language models (VLMs) with a focus on production-ready solutions.  Get started with FastDeploy on [GitHub](https://github.com/PaddlePaddle/FastDeploy).

## Key Features

*   **Production-Ready Deployment:** Out-of-the-box solutions for LLM and VLM deployment.
*   **Load-Balanced PD Disaggregation:** An industrial-grade solution that optimizes resource utilization, balances SLO compliance, and boosts throughput with context caching and dynamic instance role switching.
*   **Unified KV Cache Transmission:** High-performance transport library for efficient data transfer with intelligent NVLink/RDMA selection.
*   **OpenAI API & vLLM Compatibility:** Seamless integration with [vLLM](https://github.com/vllm-project/vllm/) interfaces for easy deployment.
*   **Comprehensive Quantization Support:** Supports various quantization formats (W8A16, W8A8, W4A16, W4A8, W2A16, FP8, etc.) for optimized model size and performance.
*   **Advanced Acceleration Techniques:** Includes speculative decoding, Multi-Token Prediction (MTP) and Chunked Prefill to accelerate inference.
*   **Multi-Hardware Support:** Compatible with NVIDIA GPUs, Kunlunxin XPU, Hygon DCU, Ascend NPU, Iluvatar GPU, Enflame GCU, MetaX GPU, and more.

## What's New

*   **[2025-09] v2.2 Release:** Hugging Face ecosystem compatibility, performance optimizations, and support for baidu/ERNIE-21B-A3B-Thinking.
*   **[2025-08] v2.1 Release:** New KV Cache scheduling strategy, expanded PD separation and CUDA Graph support, and hardware enhancements for Kunlun and Hygon.
*   **[2025-07] Inference Deployment Challenge:**  Compete for prizes by deploying ERNIE 4.5 models.  [Sign up](https://www.wjx.top/vm/meSsp3L.aspx#) and get event details ([details](https://github.com/PaddlePaddle/FastDeploy/discussions/2728)).
*   **[2025-06] v2.0 Release:** Support for ERNIE 4.5 and industrial-grade PD disaggregation.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.10 ~ 3.12

## Installation

FastDeploy supports a variety of hardware platforms. Find the installation instructions for your specific hardware:

*   [NVIDIA GPU](./docs/get_started/installation/nvidia_gpu.md)
*   [Kunlunxin XPU](./docs/get_started/installation/kunlunxin_xpu.md)
*   [Iluvatar GPU](./docs/get_started/installation/iluvatar_gpu.md)
*   [Enflame GCU](./docs/get_started/installation/Enflame_gcu.md)
*   [Hygon DCU](./docs/get_started/installation/hygon_dcu.md)
*   [MetaX GPU](./docs/get_started/installation/metax_gpu.md)

**Note:**  Support for additional hardware platforms like Ascend NPU is under active development.

## Get Started

Explore the following documentation to learn how to use FastDeploy:

*   [10-Minutes Quick Deployment](./docs/get_started/quick_start.md)
*   [ERNIE-4.5 Large Language Model Deployment](./docs/get_started/ernie-4.5.md)
*   [ERNIE-4.5-VL Multimodal Model Deployment](./docs/get_started/ernie-4.5-vl.md)
*   [Offline Inference Development](./docs/offline_inference.md)
*   [Online Service Deployment](./docs/online_serving/README.md)
*   [Best Practices](./docs/best_practices/README.md)

## Supported Models

Learn about supported models, downloading models, and using the torch format.

*   [Full Supported Models List](./docs/supported_models.md)

## Advanced Usage

*   [Quantization](./docs/quantization/README.md)
*   [PD Disaggregation Deployment](./docs/features/disaggregated.md)
*   [Speculative Decoding](./docs/features/speculative_decoding.md)
*   [Prefix Caching](./docs/features/prefix_caching.md)
*   [Chunked Prefill](./docs/features/chunked_prefill.md)

## Acknowledgement

FastDeploy is licensed under the [Apache-2.0 open-source license](./LICENSE).  We are grateful for the contributions from the [vLLM](https://github.com/vllm-project/vllm) project, which helped facilitate interface compatibility.