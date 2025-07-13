<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h1 align="center">vLLM: Fast and Efficient LLM Inference and Serving</h1>

<p align="center">
  <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a>
</p>

---

**vLLM** unlocks the potential of large language models (LLMs) by providing a high-throughput, user-friendly, and cost-effective serving solution.  <br>
**[Explore the vLLM project on GitHub](https://github.com/vllm-project/vllm)**

## Key Features

vLLM accelerates and simplifies LLM inference with these core capabilities:

*   **Blazing-Fast Inference:**
    *   State-of-the-art serving throughput.
    *   Efficient memory management with [PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html).
    *   Continuous batching of incoming requests.
    *   Optimized CUDA kernels, including integration with FlashAttention and FlashInfer.
    *   Speculative decoding and Chunked prefill for enhanced performance.
    *   Fast model execution with CUDA/HIP graph
*   **Broad Model and Hardware Support:**
    *   Seamless integration with popular Hugging Face models, including Transformers-like LLMs, Mixture-of-Expert LLMs, Embedding Models, and Multi-modal LLMs. Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).
    *   Supports NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
*   **Flexible Deployment Options:**
    *   OpenAI-compatible API server.
    *   Tensor, pipeline, data and expert parallelism support for distributed inference.
    *   Streaming outputs for real-time applications.
    *   Prefix caching support and Multi-LoRA support.
*   **Quantization Support:**
    *   Optimized with GPTQ, AWQ, AutoRound, INT4, INT8, and FP8 quantizations.
*   **Advanced Decoding Algorithms:**
    *   High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more.

## Performance

vLLM consistently outperforms other LLM serving engines. You can find performance benchmarks in our [blog post](https://blog.vllm.ai/2024/09/05/perf-update.html). The implementation is under [.buildkite/nightly-benchmarks/](.buildkite/nightly-benchmarks/) and you can [reproduce](https://github.com/vllm-project/vllm/issues/8176) this benchmark using our one-click runnable script.

## Getting Started

Install vLLM using `pip`:

```bash
pip install vllm
```

Visit our [documentation](https://docs.vllm.ai/en/latest/) for detailed information:

*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome contributions! Learn how to get involved in the [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) guide.

## Sponsors

vLLM is a community project supported by the following organizations:

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

## Contact

*   For technical questions and feature requests, use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions).
*   For discussing with fellow users, use the [vLLM Forum](https://discuss.vllm.ai).
*   For coordinating contributions and development, use [Slack](https://slack.vllm.ai).
*   For security disclosures, use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature.
*   For collaborations and partnerships, contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu).

## Media Kit

If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit).