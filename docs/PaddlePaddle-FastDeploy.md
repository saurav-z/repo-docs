# FastDeploy: Production-Ready LLM and VLM Deployment Toolkit

**FastDeploy empowers you to deploy and accelerate large language models and visual language models quickly and efficiently.**  [View the original repository on GitHub](https://github.com/PaddlePaddle/FastDeploy)

[![Python](https://img.shields.io/badge/python-3.10-aff.svg)](https://github.com/PaddlePaddle/FastDeploy)
[![OS](https://img.shields.io/badge/os-linux-pink.svg)](https://github.com/PaddlePaddle/FastDeploy)
[![Contributors](https://img.shields.io/github/contributors/PaddlePaddle/FastDeploy?color=9ea)](https://github.com/PaddlePaddle/FastDeploy/graphs/contributors)
[![Commits](https://img.shields.io/github/commit-activity/m/PaddlePaddle/FastDeploy?color=3af)](https://github.com/PaddlePaddle/FastDeploy/commits)
[![Issues](https://img.shields.io/github/issues/PaddlePaddle/FastDeploy?color=9cc)](https://github.com/PaddlePaddle/FastDeploy/issues)
[![Stars](https://img.shields.io/github/stars/PaddlePaddle/FastDeploy?color=ccf)](https://github.com/PaddlePaddle/FastDeploy/stargazers)
[![Trendshift](https://trendshift.io/api/badge/repositories/4046)](https://trendshift.io/repositories/4046)

## Key Features

*   **Accelerated Performance:** Optimized for low latency and high throughput using advanced techniques like speculative decoding and multi-token prediction.
*   **Load-Balanced PD Disaggregation:** Industrial-grade solution with context caching and dynamic instance role switching, optimizing resource utilization.
*   **Unified KV Cache Transmission:** High-performance transport library with intelligent NVLink/RDMA selection.
*   **Wide Hardware Support:**  Compatible with NVIDIA GPUs, Kunlunxin XPUs, Hygon DCUs, Ascend NPUs, and more.
*   **OpenAI API Compatibility:** Supports vLLM for seamless integration.
*   **Comprehensive Quantization:**  Supports various quantization formats, including W8A16, W8A8, W4A16, and FP8, for model size reduction and efficiency.

## News

*   **[2025-09]** FastDeploy v2.2 Released! Includes Hugging Face ecosystem model compatibility, performance enhancements, and support for [baidu/ERNIE-21B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking).
*   **[2025-08]** FastDeploy v2.1 Released! Introduces a new KV Cache scheduling strategy, expanded PD separation and CUDA Graph support, and enhanced hardware support, including Kunlun and Hygon platforms.
*   **[2025-07]** FastDeploy 2.0 Inference Deployment Challenge is Live!

## About

**FastDeploy** is a powerful toolkit designed for deploying and accelerating Large Language Models (LLMs) and Visual Language Models (VLMs) based on PaddlePaddle. FastDeploy provides a full stack solution.

## Requirements

*   OS: Linux
*   Python: 3.10 ~ 3.12

## Installation

FastDeploy supports inference deployment on various hardware platforms.

-   [NVIDIA GPU](./docs/get_started/installation/nvidia_gpu.md)
-   [Kunlunxin XPU](./docs/get_started/installation/kunlunxin_xpu.md)
-   [Iluvatar GPU](./docs/get_started/installation/iluvatar_gpu.md)
-   [Enflame GCU](./docs/get_started/installation/Enflame_gcu.md)
-   [Hygon DCU](./docs/get_started/installation/hygon_dcu.md)
-   [MetaX GPU](./docs/get_started/installation/metax_gpu.md)

## Get Started

Explore the following resources to begin using FastDeploy:

*   [10-Minutes Quick Deployment](./docs/get_started/quick_start.md)
*   [ERNIE-4.5 Large Language Model Deployment](./docs/get_started/ernie-4.5.md)
*   [ERNIE-4.5-VL Multimodal Model Deployment](./docs/get_started/ernie-4.5-vl.md)
*   [Offline Inference Development](./docs/offline_inference.md)
*   [Online Service Deployment](./docs/online_serving/README.md)
*   [Best Practices](./docs/best_practices/README.md)

## Supported Models

*   [Full Supported Models List](./docs/supported_models.md)

## Advanced Usage

*   [Quantization](./docs/quantization/README.md)
*   [PD Disaggregation Deployment](./docs/features/disaggregated.md)
*   [Speculative Decoding](./docs/features/speculative_decoding.md)
*   [Prefix Caching](./docs/features/prefix_caching.md)
*   [Chunked Prefill](./docs/features/chunked_prefill.md)

## Acknowledgement

FastDeploy is licensed under the [Apache-2.0 open-source license](./LICENSE). Parts of [vLLM](https://github.com/vllm-project/vllm) were used to maintain interface compatibility.