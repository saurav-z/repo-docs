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

# FastDeploy: Accelerate Your LLM and VLM Deployment with Production-Ready Solutions

**FastDeploy** streamlines the deployment of large language models (LLMs) and visual language models (VLMs), offering a high-performance, production-ready toolkit based on PaddlePaddle. [Explore the FastDeploy GitHub Repository](https://github.com/PaddlePaddle/FastDeploy) for more details.

## Key Features

*   **Optimized Performance:** Benefit from core acceleration technologies for fast and efficient LLM/VLM inference and deployment.
    *   🚀 **Load-Balanced PD Disaggregation:** Industrial-grade solution featuring context caching and dynamic instance role switching. Optimizes resource utilization while balancing SLO compliance and throughput.
    *   🔄 **Unified KV Cache Transmission:** Lightweight high-performance transport library with intelligent NVLink/RDMA selection.
    *   🤝 **OpenAI API Server and vLLM Compatible:** One-command deployment with [vLLM](https://github.com/vllm-project/vllm/) interface compatibility.
    *   🧮 **Comprehensive Quantization Format Support:** W8A16, W8A8, W4A16, W4A8, W2A16, FP8, and more.
    *   ⏩ **Advanced Acceleration Techniques:** Speculative decoding, Multi-Token Prediction (MTP) and Chunked Prefill.
    *   🖥️ **Multi-Hardware Support:** NVIDIA GPU, Kunlunxin XPU, Hygon DCU, Ascend NPU, Iluvatar GPU, Enflame GCU, MetaX GPU etc.

## What's New

*   **v2.2 (2025-09):**  Compatibility with HuggingFace models, further performance optimizations, and support for [baidu/ERNIE-21B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking).
*   **v2.1 (2025-08):**  New KV Cache scheduling strategy, expanded PD separation and CUDA Graph support, and enhanced hardware support for platforms like Kunlun and Hygon.
*   **v2.0 (2025-06):**  Support for ERNIE 4.5 inference and deployment, and an industrial-grade PD disaggregation with context caching.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.10 - 3.12

## Installation

Install FastDeploy for your preferred hardware:

*   [NVIDIA GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/)
*   [Kunlunxin XPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/kunlunxin_xpu/)
*   [Iluvatar GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/iluvatar_gpu/)
*   [Enflame GCU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/Enflame_gcu/)
*   [Hygon DCU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/hygon_dcu/)
*   [MetaX GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/metax_gpu/)

**Note:**  Support for Ascend NPU and other hardware platforms is under active development.

## Get Started

Quickly deploy and utilize FastDeploy with the following resources:

*   [10-Minute Quick Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/quick_start)
*   [ERNIE-4.5 Large Language Model Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/ernie-4.5.md)
*   [ERNIE-4.5-VL Multimodal Model Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/ernie-4.5-vl.md)
*   [Offline Inference Development](https://paddlepaddle.github.io/FastDeploy/offline_inference.md)
*   [Online Service Deployment](https://paddlepaddle.github.io/FastDeploy/online_serving/README.md)
*   [Best Practices](https://paddlepaddle.github.io/FastDeploy/best_practices/README.md)

## Supported Models

*   [Full Supported Models List](https://paddlepaddle.github.io/FastDeploy/supported_models.md)

## Advanced Usage

*   [Quantization](https://paddlepaddle.github.io/FastDeploy/quantization/README.md)
*   [PD Disaggregation Deployment](https://paddlepaddle.github.io/FastDeploy/features/disaggregated.md)
*   [Speculative Decoding](https://paddlepaddle.github.io/FastDeploy/features/speculative_decoding.md)
*   [Prefix Caching](https://paddlepaddle.github.io/FastDeploy/features/prefix_caching.md)
*   [Chunked Prefill](https://paddlepaddle.github.io/FastDeploy/features/chunked_prefill.md)

## Acknowledgements

FastDeploy is licensed under the [Apache-2.0 open-source license](./LICENSE).  We gratefully acknowledge the contributions of the [vLLM](https://github.com/vllm-project/vllm) project.