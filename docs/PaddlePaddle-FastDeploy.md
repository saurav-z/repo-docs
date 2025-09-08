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
  <a href="https://trendshift.io/repositories/4046" target="_blank"><img src="https://trendshift.io/api/badge/repositories/4046" alt="PaddlePaddle%2FFastDeploy | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
  </br>
  <a href="https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/"><b> Installation </b></a> |
  <a href="https://paddlepaddle.github.io/FastDeploy/get_started/quick_start"><b> Quick Start </b></a> |
  <a href="https://paddlepaddle.github.io/FastDeploy/supported_models/"><b> Supported Models </b></a>
</p>

--------------------------------------------------------------------------------

# FastDeploy: Accelerate Your LLMs and VLMs with Production-Ready Deployment

**FastDeploy**, developed by PaddlePaddle, provides a comprehensive toolkit to effortlessly deploy large language models (LLMs) and visual language models (VLMs) into production environments.  [Explore the original repository](https://github.com/PaddlePaddle/FastDeploy).

## Key Features

*   **🚀 Production-Ready Deployment:** Out-of-the-box solutions for LLM and VLM deployment.
*   **⚡ Load-Balanced PD Disaggregation:** Industrial-grade solution with context caching and dynamic instance role switching for optimized resource utilization.
*   **🔄 Unified KV Cache Transmission:** High-performance transport with intelligent NVLink/RDMA selection.
*   **🤝 OpenAI API Server & vLLM Compatible:** Deploy with a single command and integrate with vLLM interfaces.
*   **🧮 Comprehensive Quantization Support:** Supports W8A16, W8A8, W4A16, W4A8, W2A16, FP8, and more for model optimization.
*   **⏩ Advanced Acceleration Techniques:** Includes speculative decoding, Multi-Token Prediction (MTP), and Chunked Prefill for superior performance.
*   **🖥️ Multi-Hardware Support:** Optimized for NVIDIA GPUs, Kunlunxin XPU, Hygon DCU, Ascend NPU, Iluvatar GPU, Enflame GCU, MetaX GPU and more.

## What's New

*   **[2024-08] 🔥 Released FastDeploy v2.1:** Introduces a brand-new KV Cache scheduling strategy, expanded support for PD separation and CUDA Graph, and enhanced hardware support.
*   **[2024-07] 🔥 FastDeploy 2.0 Inference Deployment Challenge:** Participate in the ERNIE 4.5 series deployment challenge and win prizes! [Sign up here](https://www.wjx.top/vm/meSsp3L.aspx#) [Event details](https://github.com/PaddlePaddle/FastDeploy/discussions/2728)
*   **[2024-06] 🔥 Released FastDeploy v2.0:** Includes support for ERNIE 4.5 and open-sources industrial-grade PD disaggregation.

## Requirements

*   OS: Linux
*   Python: 3.10 - 3.12

## Installation

FastDeploy supports a variety of hardware for efficient inference deployment.  See the detailed installation instructions below for your specific hardware:

*   [NVIDIA GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu.md)
*   [Kunlunxin XPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/kunlunxin_xpu.md)
*   [Iluvatar GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/iluvatar_gpu.md)
*   [Enflame GCU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/Enflame_gcu.md)
*   [Hygon DCU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/hygon_dcu.md)

**Note:**  Support for Ascend NPU and MetaX GPU is actively under development.

## Get Started

Quickly begin using FastDeploy with these helpful resources:

*   [10-Minute Quick Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/quick_start.md)
*   [ERNIE-4.5 Large Language Model Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/ernie-4.5.md)
*   [ERNIE-4.5-VL Multimodal Model Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/ernie-4.5-vl.md)
*   [Offline Inference Development](https://paddlepaddle.github.io/FastDeploy/docs/offline_inference.md)
*   [Online Service Deployment](https://paddlepaddle.github.io/FastDeploy/docs/online_serving/README.md)
*   [Full Supported Models List](https://paddlepaddle.github.io/FastDeploy/docs/supported_models.md)
*   [Best Practices](https://paddlepaddle.github.io/FastDeploy/docs/best_practices/README.md)

## Supported Models

| Model                       | Data Type                                     | PD Disaggregation | Chunked Prefill | Prefix Caching |  MTP | CUDA Graph | Maximum Context Length |
| :-------------------------- | :--------------------------------------------- | :---------------- | :-------------- | :------------- | :--- | :--------- | :--------------------- |
| ERNIE-4.5-300B-A47B          | BF16/WINT4/WINT8/W4A8C8/WINT2/FP8             | ✅                | ✅              | ✅             | ✅   | ✅         | 128K                   |
| ERNIE-4.5-300B-A47B-Base     | BF16/WINT4/WINT8                               | ✅                | ✅              | ✅             | ❌   | ✅         | 128K                   |
| ERNIE-4.5-VL-424B-A47B       | BF16/WINT4/WINT8                               | WIP               | ✅              | WIP            | ❌   | WIP        | 128K                   |
| ERNIE-4.5-VL-28B-A3B         | BF16/WINT4/WINT8                               | ❌                | ✅              | WIP            | ❌   | WIP        | 128K                   |
| ERNIE-4.5-21B-A3B            | BF16/WINT4/WINT8/FP8                           | ❌                | ✅              | ✅             | ✅   | ✅         | 128K                   |
| ERNIE-4.5-21B-A3B-Base       | BF16/WINT4/WINT8/FP8                           | ✅                | ✅              | ✅             | ❌   | ✅         | 128K                   |
| ERNIE-4.5-0.3B               | BF16/WINT8/FP8                                 | ✅                | ✅              | ✅             | ❌   | ✅         | 128K                   |

## Advanced Usage

Explore advanced features to optimize your model deployment:

*   [Quantization](https://paddlepaddle.github.io/FastDeploy/docs/quantization/README.md)
*   [PD Disaggregation Deployment](https://paddlepaddle.github.io/FastDeploy/docs/features/disaggregated.md)
*   [Speculative Decoding](https://paddlepaddle.github.io/FastDeploy/docs/features/speculative_decoding.md)
*   [Prefix Caching](https://paddlepaddle.github.io/FastDeploy/docs/features/prefix_caching.md)
*   [Chunked Prefill](https://paddlepaddle.github.io/FastDeploy/docs/features/chunked_prefill.md)

## Acknowledgement

FastDeploy is licensed under the [Apache-2.0 open-source license](./LICENSE). Parts of the [vLLM](https://github.com/vllm-project/vllm) code were used and incorporated, and we thank them for their contribution.