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

# FastDeploy: Deploy LLMs and VLMs with Speed and Efficiency

**FastDeploy empowers developers to deploy Large Language Models (LLMs) and Visual Language Models (VLMs) with optimized performance and production-ready features.** ([View the original repository](https://github.com/PaddlePaddle/FastDeploy))

## Key Features:

*   **Optimized Inference:** Accelerates LLM and VLM inference with advanced techniques.
*   **Production-Ready Deployment:** Provides out-of-the-box deployment solutions.
*   **Load-Balanced PD Disaggregation**: Industrial-grade solution featuring context caching and dynamic instance role switching. Optimizes resource utilization while balancing SLO compliance and throughput.
*   **Unified KV Cache Transmission:** Lightweight high-performance transport library with intelligent NVLink/RDMA selection.
*   **OpenAI API & vLLM Compatibility:** Seamless integration with the vLLM interface for easy deployment.
*   **Comprehensive Quantization Support:** Offers extensive quantization options (W8A16, W8A8, W4A16, W4A8, W2A16, FP8, etc.) for model size reduction and performance gains.
*   **Advanced Acceleration Techniques:** Implements speculative decoding, Multi-Token Prediction (MTP), and Chunked Prefill for faster inference.
*   **Multi-Hardware Support:** Runs efficiently on various hardware platforms, including NVIDIA GPUs, Kunlunxin XPUs, Hygon DCUs, Ascend NPUs, and more.

## News

*   **[2025-09] 🔥 FastDeploy v2.2 is newly released!** It now offers compatibility with models in the HuggingFace ecosystem, has further optimized performance, and newly adds support for [baidu/ERNIE-21B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking)!
*   **[2025-08] 🔥 Released FastDeploy v2.1:** A brand-new KV Cache scheduling strategy has been introduced, and expanded support for PD separation and CUDA Graph across more models. Enhanced hardware support has been added for platforms like Kunlun and Hygon, along with comprehensive optimizations to improve the performance of both the service and inference engine.
*   **[2025-07] The FastDeploy 2.0 Inference Deployment Challenge is now live!** Complete the inference deployment task for the ERNIE 4.5 series open-source models to win official FastDeploy 2.0 merch and generous prizes! 🎁 You're welcome to try it out and share your feedback! 📌[Sign up here](https://www.wjx.top/vm/meSsp3L.aspx#) 📌[Event details](https://github.com/PaddlePaddle/FastDeploy/discussions/2728)
*   **[2025-06] 🔥 Released FastDeploy v2.0:** Supports inference and deployment for ERNIE 4.5. Furthermore, we open-source an industrial-grade PD disaggregation with context caching, dynamic role switching for effective resource utilization to further enhance inference performance for MoE models.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.10 - 3.12

## Installation

FastDeploy supports deployment on various hardware platforms. Choose your platform:

*   [NVIDIA GPU](./docs/get_started/installation/nvidia_gpu.md)
*   [Kunlunxin XPU](./docs/get_started/installation/kunlunxin_xpu.md)
*   [Iluvatar GPU](./docs/get_started/installation/iluvatar_gpu.md)
*   [Enflame GCU](./docs/get_started/installation/Enflame_gcu.md)
*   [Hygon DCU](./docs/get_started/installation/hygon_dcu.md)
*   [MetaX GPU](./docs/get_started/installation/metax_gpu.md)

**Note:** We are actively expanding hardware support, including Ascend NPU. Stay tuned for updates!

## Get Started

Explore the documentation:

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

FastDeploy is licensed under the [Apache-2.0 open-source license](./LICENSE).  We acknowledge the contributions of [vLLM](https://github.com/vllm-project/vllm) and express our gratitude for their code used in maintaining interface compatibility.
```
Key improvements and explanations:

*   **SEO Optimization:**
    *   Included relevant keywords in the title, headings, and descriptions (e.g., "LLMs," "VLMs," "inference," "deployment").
    *   Used clear and concise language.
    *   Added a descriptive title and introductory sentence.

*   **Structure and Readability:**
    *   Used headings and subheadings to organize the content logically.
    *   Employed bullet points for key features to improve scannability.
    *   Simplified the "About" section and condensed the information.

*   **Content Enhancement:**
    *   Added a strong one-sentence hook to grab the reader's attention.
    *   Clarified the benefits of FastDeploy (speed, efficiency, production-readiness).
    *   Made the installation section clearer.
    *   Improved the overall flow and readability of the README.

*   **Conciseness:**
    *   Removed redundant information.
    *   Focused on the most important details.

*   **Actionable Information:**
    *   Provided links to the original repo and relevant documentation.
    *   Made it easy for users to get started.