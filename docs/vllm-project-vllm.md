---
title: vLLM: Fast and Efficient LLM Inference and Serving - [Project Name]
description: vLLM is a fast and easy-to-use library for LLM inference and serving, enabling efficient and cost-effective deployment of large language models.
keywords: vLLM, LLM, large language models, inference, serving, PagedAttention, CUDA, Hugging Face, OpenAI, distributed inference, quantization, model serving
---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h2 align="center">vLLM: Revolutionizing LLM Serving for Speed and Efficiency</h2>

<p align="center">
  <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
  <a href="https://github.com/vllm-project/vllm"><b>GitHub</b></a>
</p>

---

vLLM empowers you to serve LLMs faster, cheaper, and easier than ever before.  Developed at UC Berkeley and now a thriving community project, vLLM is engineered for high-performance LLM inference and serving.

**Key Features of vLLM:**

*   **Blazing Fast Inference:**
    *   State-of-the-art serving throughput.
    *   Optimized with PagedAttention for efficient memory management.
    *   Continuous batching for handling incoming requests.
    *   Fast model execution with CUDA/HIP graph.
    *   Speculative Decoding and Chunked Prefill
*   **Flexible and Easy to Use:**
    *   Seamless Hugging Face integration.
    *   Supports various decoding algorithms (parallel sampling, beam search, etc.).
    *   Distributed inference with tensor, pipeline, data, and expert parallelism.
    *   Streaming output for real-time results.
    *   OpenAI-compatible API server.
    *   Supports NVIDIA, AMD, Intel, PowerPC, TPU, and AWS Neuron.
    *   Multi-LoRA and Prefix Caching support
*   **Broad Model Support:**
    *   Supports most popular open-source models on Hugging Face including:
        *   Transformer-like LLMs (e.g., Llama)
        *   Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
        *   Embedding Models (e.g., E5-Mistral)
        *   Multi-modal LLMs (e.g., LLaVA)
        *   Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).
*   **Optimized for Cost Efficiency:**
    *   Quantization support: GPTQ, AWQ, AutoRound, INT4, INT8, and FP8

## Getting Started

Install vLLM easily using pip:

```bash
pip install vllm
```

Or build from source: [Build from Source Instructions](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source).

Explore our comprehensive [documentation](https://docs.vllm.ai/en/latest/) for more in-depth information:

*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome contributions! Learn how to contribute to the project at: [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html).

## Sponsors

vLLM is a community project and is supported by many organizations:

**Cash Donations:**

*   a16z
*   Dropbox
*   Sequoia Capital
*   Skywork AI
*   ZhenFund

**Compute Resources:**

*   AMD
*   Anyscale
*   AWS
*   Crusoe Cloud
*   Databricks
*   DeepInfra
*   Google Cloud
*   Intel
*   Lambda Lab
*   Nebius
*   Novita AI
*   NVIDIA
*   Replicate
*   Roblox
*   RunPod
*   Trainy
*   UC Berkeley
*   UC San Diego

**Slack Sponsor:** Anyscale

Support the project via [OpenCollective](https://opencollective.com/vllm)

## Citation

If you use vLLM in your research, please cite our paper:

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

*   For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
*   For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
*   For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
*   For collaborations and partnerships, please contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

## Media Kit

Find the vLLM logo and branding assets in the [media kit repo](https://github.com/vllm-project/media-kit).
```
Key improvements and SEO optimizations:

*   **Clear Title and Description:**  Includes the target keywords and a concise description optimized for search engines.
*   **SEO Keywords:** Added relevant keywords in the description to boost search visibility.
*   **Structured Headings:**  Uses `<h2>` and `<h3>` tags for better readability and SEO.
*   **Concise Language:**  Improved the phrasing and overall structure of the text.
*   **Call to Action:**  Highlights the benefits of using vLLM.
*   **Emphasis on Benefits:** Focused on what users gain (speed, efficiency, cost savings).
*   **Internal Linking:** Added links to key sections.
*   **GitHub Link:** Included the link to the original repository.
*   **Community Focus:** Highlighted that the project is community-driven.
*   **Clear Structure:** Added a "Key Features" section with a bulleted list for quick understanding.
*   **Removed Redundancy:** Eliminated repetitive phrases.
*   **Maintain Original Content:**  Incorporated the original content while improving structure and clarity.