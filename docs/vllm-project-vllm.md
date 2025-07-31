<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h1 align="center">vLLM: Fast and Efficient LLM Serving</h1>

<p align="center">
  <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
  <br>
  <a href="https://github.com/vllm-project/vllm"><b>View on GitHub</b></a>
</p>

---

**vLLM empowers you to serve large language models (LLMs) quickly, easily, and affordably, unlocking the potential of AI for everyone.**

## Key Features

*   **Blazing-Fast Inference:** Achieve state-of-the-art serving throughput with optimized CUDA kernels, including FlashAttention and FlashInfer.
*   **Memory Efficiency:** Leverage PagedAttention for efficient management of attention key and value memory.
*   **Continuous Batching:** Optimize throughput by continuously batching incoming requests.
*   **Quantization Support:** Reduce memory footprint and accelerate inference with support for GPTQ, AWQ, AutoRound, INT4, INT8, and FP8.
*   **Flexible Deployment:**  Includes an OpenAI-compatible API server, streaming outputs, and supports tensor, pipeline, data, and expert parallelism.
*   **Broad Hardware Compatibility:** Supports NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
*   **Seamless Model Integration:** Works with most popular open-source models on Hugging Face, including Transformers, Mixture-of-Experts, and Multi-modal LLMs.
*   **Advanced Decoding Algorithms:** Offers high-throughput serving with parallel sampling, beam search, and more.
*   **Prefix Caching & Multi-LoRA Support:** Enhanced functionality to further optimize inference performance.

## About vLLM

vLLM, originally developed at UC Berkeley's Sky Computing Lab, is a community-driven project dedicated to making LLM inference and serving accessible and efficient.  It provides a streamlined solution for deploying and utilizing LLMs with optimized performance and resource utilization.  It allows everyone to use LLMs, from researchers to businesses.

## Getting Started

Easily install vLLM using pip:

```bash
pip install vllm
```

Or, [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for more advanced options.

Explore the following resources to learn more:

*   [Installation Guide](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart Tutorial](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome contributions! Check out the [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) guide to get involved.

## Sponsors

vLLM is supported by a community of organizations.  Thank you to our sponsors!

<!-- Note: Please sort them in alphabetical order. -->
<!-- Note: Please keep these consistent with docs/community/sponsors.md -->
Cash Donations:

- a16z
- Dropbox
- Sequoia Capital
- Skywork AI
- ZhenFund

Compute Resources:

- AMD
- Anyscale
- AWS
- Crusoe Cloud
- Databricks
- DeepInfra
- Google Cloud
- Intel
- Lambda Lab
- Nebius
- Novita AI
- NVIDIA
- Replicate
- Roblox
- RunPod
- Trainy
- UC Berkeley
- UC San Diego

Slack Sponsor: Anyscale

We also have an official fundraising venue through [OpenCollective](https://opencollective.com/vllm). We plan to use the fund to support the development, maintenance, and adoption of vLLM.

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

-   For technical questions and feature requests, use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
-   Join the [vLLM Forum](https://discuss.vllm.ai) for discussions with fellow users
-   Coordinate contributions and development on [Slack](https://slack.vllm.ai)
-   Report security disclosures via GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories)
-   For collaborations and partnerships, contact [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

## Media Kit

For vLLM logo usage, please refer to our [media kit repo](https://github.com/vllm-project/media-kit).