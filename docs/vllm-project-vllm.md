<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h1 align="center">vLLM: Revolutionizing LLM Serving with Speed and Efficiency</h1>

<p align="center">
  <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
  <br>
  <a href="https://github.com/vllm-project/vllm"><b>View on GitHub</b></a>
</p>

---

vLLM empowers everyone to serve Large Language Models (LLMs) easily, quickly, and affordably.  Developed at UC Berkeley and now a community-driven project, vLLM offers state-of-the-art performance for LLM inference.

## Key Features

*   **Blazing Fast Inference:**  Achieve top-tier serving throughput with optimized CUDA kernels, continuous batching, and efficient memory management.
*   **Efficient Memory Management:**  Leverages **PagedAttention** for efficient attention key and value memory handling.
*   **Broad Hardware Support:** Runs on NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPUs, and AWS Neuron.
*   **Quantization:** Supports various quantization techniques, including GPTQ, AWQ, AutoRound, INT4, INT8, and FP8, for reduced memory footprint.
*   **Flexible Decoding Algorithms:** Supports parallel sampling, beam search, and other algorithms.
*   **Model Compatibility:** Seamlessly integrates with a wide range of Hugging Face models, including transformer-like LLMs (e.g., Llama), Mixture-of-Experts LLMs (e.g., Mixtral), embedding models (e.g., E5-Mistral), and multi-modal LLMs (e.g., LLaVA).
*   **Distributed Inference:**  Supports tensor, pipeline, data, and expert parallelism for distributed inference.
*   **API Compatibility:** Offers an OpenAI-compatible API server for easy integration.
*   **Streaming & Prefix Caching:** Provides streaming outputs and prefix caching support for enhanced performance.
*   **Multi-LoRA Support:** Supports multi-LoRA to help with serving many fine-tuned models efficiently.
*   **Speculative Decoding:** Provides speculative decoding to generate responses faster.
*   **Chunked Prefill:** Utilizes chunked prefill to reduce latency.

## Getting Started

Install vLLM using pip:

```bash
pip install vllm
```

Comprehensive documentation is available to guide you through the process.  Explore the [documentation](https://docs.vllm.ai/en/latest/) for:

*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

vLLM is an open-source project, and contributions are highly valued. See [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for details on how to participate.

## Sponsors

vLLM is supported by the following organizations:

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

*   For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
*   For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
*   For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
*   For collaborations and partnerships, please contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)