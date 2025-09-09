# FastDeploy: Supercharge Your LLM and VLM Deployment with Production-Ready Solutions

**FastDeploy** is your go-to toolkit for effortlessly deploying Large Language Models (LLMs) and Visual Language Models (VLMs) built on PaddlePaddle, offering production-ready deployment solutions with cutting-edge acceleration technologies. ([Original Repo](https://github.com/PaddlePaddle/FastDeploy))

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

## Key Features

*   **Optimized Performance:** Achieve high throughput and low latency with advanced acceleration techniques and optimized KV Cache handling.
*   **Comprehensive Hardware Support:** Deploy on NVIDIA GPUs, Kunlunxin XPUs, Hygon DCUs, Ascend NPUs, Iluvatar GPUs, Enflame GCUs, and MetaX GPUs.
*   **Model Compatibility:** Supports a wide range of quantization formats including W8A16, W8A8, W4A16, W4A8, W2A16, and FP8, as well as models from HuggingFace.
*   **Production-Ready Deployment:** Benefit from features like load-balanced PD disaggregation, OpenAI API server and vLLM compatibility, and advanced acceleration techniques like speculative decoding.

## News

*   **[2025-09] 🔥 FastDeploy v2.2 is newly released!** Compatibility with models in the HuggingFace ecosystem, further optimized performance, and newly adds support for [baidu/ERNIE-21B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking)!
*   **[2025-08] 🔥 Released FastDeploy v2.1:** A brand-new KV Cache scheduling strategy has been introduced, and expanded support for PD separation and CUDA Graph across more models. Enhanced hardware support has been added for platforms like Kunlun and Hygon, along with comprehensive optimizations to improve the performance of both the service and inference engine.
*   **[2025-07] The FastDeploy 2.0 Inference Deployment Challenge is now live!** Complete the inference deployment task for the ERNIE 4.5 series open-source models to win official FastDeploy 2.0 merch and generous prizes! 🎁 You're welcome to try it out and share your feedback! 📌[Sign up here](https://www.wjx.top/vm/meSsp3L.aspx#) 📌[Event details](https://github.com/PaddlePaddle/FastDeploy/discussions/2728)
*   **[2025-06] 🔥 Released FastDeploy v2.0:** Supports inference and deployment for ERNIE 4.5. Furthermore, we open-source an industrial-grade PD disaggregation with context caching, dynamic role switching for effective resource utilization to further enhance inference performance for MoE models.

## Requirements

*   OS: Linux
*   Python: 3.10 ~ 3.12

## Installation

FastDeploy supports inference deployment on **NVIDIA GPUs**, **Kunlunxin XPUs**, **Iluvatar GPUs**, **Enflame GCUs**, **Hygon DCUs** and other hardware. For detailed installation instructions:

*   [NVIDIA GPU](./docs/get_started/installation/nvidia_gpu.md)
*   [Kunlunxin XPU](./docs/get_started/installation/kunlunxin_xpu.md)
*   [Iluvatar GPU](./docs/get_started/installation/iluvatar_gpu.md)
*   [Enflame GCU](./docs/get_started/installation/Enflame_gcu.md)
*   [Hygon DCU](./docs/get_started/installation/hygon_dcu.md)
*   [MetaX GPU](./docs/get_started/installation/metax_gpu.md)

**Note:** We are actively working on expanding hardware support. Additional hardware platforms including Ascend NPU are currently under development and testing. Stay tuned for updates!

## Get Started

Explore FastDeploy with these resources:

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

FastDeploy is licensed under the [Apache-2.0 open-source license](./LICENSE). During development, portions of [vLLM](https://github.com/vllm-project/vllm) code were referenced and incorporated to maintain interface compatibility, for which we express our gratitude.