# FastDeploy: Supercharge Your LLM and VLM Deployment with PaddlePaddle

**FastDeploy simplifies and accelerates the deployment of large language models (LLMs) and visual language models (VLMs), offering production-ready solutions optimized for performance.**

[![Python 3.10](https://img.shields.io/badge/python-3.10-aff.svg)](https://www.python.org/)
[![OS: Linux](https://img.shields.io/badge/os-linux-pink.svg)](https://www.linux.org/)
[![Contributors](https://img.shields.io/github/contributors/PaddlePaddle/FastDeploy?color=9ea)](https://github.com/PaddlePaddle/FastDeploy/graphs/contributors)
[![Commits](https://img.shields.io/github/commit-activity/m/PaddlePaddle/FastDeploy?color=3af)](https://github.com/PaddlePaddle/FastDeploy/commits)
[![Issues](https://img.shields.io/github/issues/PaddlePaddle/FastDeploy?color=9cc)](https://github.com/PaddlePaddle/FastDeploy/issues)
[![Stars](https://img.shields.io/github/stars/PaddlePaddle/FastDeploy?color=ccf)](https://github.com/PaddlePaddle/FastDeploy/stargazers)

<p align="center">
    <a href="https://trendshift.io/repositories/4046" target="_blank"><img src="https://trendshift.io/api/badge/repositories/4046" alt="PaddlePaddle%2FFastDeploy | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

[Installation](https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/)
| [Quick Start](https://paddlepaddle.github.io/FastDeploy/get_started/quick_start)
| [Supported Models](https://paddlepaddle.github.io/FastDeploy/supported_models/)

**[Explore the original repository](https://github.com/PaddlePaddle/FastDeploy)**

## Key Features

*   **Optimized Performance:** Leverage advanced acceleration techniques for faster inference.
*   **Production-Ready Deployments:** Get out-of-the-box solutions for LLMs and VLMs.
*   **Hardware Agnostic**: Supports NVIDIA GPU, Kunlunxin XPU, Hygon DCU, Ascend NPU, Iluvatar GPU, Enflame GCU, MetaX GPU, Intel Gaudi and more.
*   **Load-Balanced PD Disaggregation:** Achieve optimal resource utilization with context caching and dynamic instance role switching.
*   **Comprehensive Quantization:** Supports W8A16, W8A8, W4A16, W4A8, W2A16, FP8, and more.
*   **OpenAI API & vLLM Compatibility:** Easily deploy with vLLM interface compatibility.
*   **Advanced Acceleration Techniques**: Speculative decoding, Multi-Token Prediction (MTP) and Chunked Prefill.

## What's New

*   **[2025-09] v2.2 Release:** Compatibility with HuggingFace models, performance optimizations, and support for baidu/ERNIE-21B-A3B-Thinking.
*   **[2025-08] v2.1 Release:** Introduced a new KV Cache scheduling strategy, extended support for PD separation and CUDA Graph, enhanced hardware support for Kunlun and Hygon, and comprehensive optimizations.
*   **[2025-07] Inference Deployment Challenge:** Participate in the FastDeploy 2.0 Challenge for ERNIE 4.5 and win prizes!

## About FastDeploy

FastDeploy is a powerful inference and deployment toolkit designed for accelerating the deployment of large language models (LLMs) and visual language models (VLMs) using PaddlePaddle. It provides production-ready deployment solutions with core acceleration technologies and advanced features to boost performance, reduce latency, and maximize resource utilization across various hardware platforms.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.10 ~ 3.12

## Installation Guide

FastDeploy supports deployment on various hardware platforms:

*   [NVIDIA GPU](docs/get_started/installation/nvidia_gpu.md)
*   [Kunlunxin XPU](docs/get_started/installation/kunlunxin_xpu.md)
*   [Iluvatar GPU](docs/get_started/installation/iluvatar_gpu.md)
*   [Enflame GCU](docs/get_started/installation/Enflame_gcu.md)
*   [Hygon DCU](docs/get_started/installation/hygon_dcu.md)
*   [MetaX GPU](docs/get_started/installation/metax_gpu.md)
*   [Intel Gaudi](docs/get_started/installation/intel_gaudi.md)

**Note:** Support for additional hardware platforms is continuously expanding.

## Get Started

Quickly get up and running with FastDeploy using these resources:

*   [10-Minutes Quick Deployment](docs/get_started/quick_start.md)
*   [ERNIE-4.5 Large Language Model Deployment](docs/get_started/ernie-4.5.md)
*   [ERNIE-4.5-VL Multimodal Model Deployment](docs/get_started/ernie-4.5-vl.md)
*   [Offline Inference Development](docs/offline_inference.md)
*   [Online Service Deployment](docs/online_serving/README.md)
*   [Best Practices](docs/best_practices/README.md)

## Supported Models

Discover the extensive list of supported models:

*   [Full Supported Models List](docs/supported_models.md)

## Advanced Usage

Explore advanced features and capabilities:

*   [Quantization](docs/quantization/README.md)
*   [PD Disaggregation Deployment](docs/features/disaggregated.md)
*   [Speculative Decoding](docs/features/speculative_decoding.md)
*   [Prefix Caching](docs/features/prefix_caching.md)
*   [Chunked Prefill](docs/features/chunked_prefill.md)

## Acknowledgement

FastDeploy is licensed under the [Apache-2.0 open-source license](./LICENSE). We acknowledge and appreciate the portions of code from [vLLM](https://github.com/vllm-project/vllm) that were referenced and incorporated during development.