<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h1 align="center">vLLM: Fast and Efficient LLM Serving</h1>

<p align="center">
  <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a>
</p>

---

## About vLLM

**vLLM is a groundbreaking library designed to make serving large language models (LLMs) fast, easy, and cost-effective for everyone.**  Developed at UC Berkeley and now a thriving community project, vLLM empowers you to run LLMs with unparalleled efficiency.  [Explore the original repo](https://github.com/vllm-project/vllm) to get started.

**Key Features:**

*   🚀 **High-Throughput Serving:** Achieve state-of-the-art throughput for LLM serving.
*   🧠 **PagedAttention:**  Efficiently manages attention key and value memory for optimized performance.
*   🔄 **Continuous Batching:**  Dynamically batch incoming requests to maximize GPU utilization.
*   ⚡ **Fast Model Execution:**  Leverages CUDA/HIP graphs for rapid model execution.
*   ⚙️ **Quantization Support:**  Offers a wide range of quantization methods like GPTQ, AWQ, INT4, INT8, and FP8 to reduce memory footprint and improve speed.
*   💡 **Optimized Kernels:**  Includes optimized CUDA kernels, with integrations like FlashAttention and FlashInfer.
*   ✨ **Advanced Decoding:** Supports speculative decoding and chunked prefill.
*   🧩 **Seamless Integration:** Works smoothly with popular Hugging Face models.
*   🌐 **Distributed Inference:** Supports tensor and pipeline parallelism for scaling up inference.
*   📡 **Streaming Outputs:**  Provides real-time streaming of generated text.
*   💻 **OpenAI-Compatible API:**  Offers an OpenAI-compatible API server for easy integration.
*   ⚙️ **Hardware and Software Support:**  Supports NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
*   💾 **Prefix Caching:** Includes prefix caching support.
*   🎭 **Multi-LoRA Support:** Offers multi-LoRA support

vLLM supports a vast array of open-source models, including:
*   Transformer-like LLMs (e.g., Llama)
*   Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
*   Embedding Models (e.g., E5-Mistral)
*   Multi-modal LLMs (e.g., LLaVA)

  Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

**Installation:**

Install vLLM using pip:

```bash
pip install vllm
```

Or, build from [source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source).

**Resources:**

*   [Documentation](https://docs.vllm.ai/en/latest/)
    *   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
    *   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
    *   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Performance Benchmarks

vLLM consistently outperforms other LLM serving engines in terms of throughput. See the performance benchmark at the end of [our blog post](https://blog.vllm.ai/2024/09/05/perf-update.html). The implementation is under [.buildkite/nightly-benchmarks/](.buildkite/nightly-benchmarks/) and you can [reproduce](https://github.com/vllm-project/vllm/issues/8176) this benchmark using our one-click runnable script.

## Contributing

We welcome contributions!  Check out the [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) guide to get involved.

## Sponsors

vLLM is a community project supported by generous sponsors.  A complete list is available in [docs/community/sponsors.md](https://github.com/vllm-project/vllm/blob/main/docs/community/sponsors.md).

*   **Cash Donations:** a16z, Dropbox, Sequoia Capital, Skywork AI, ZhenFund
*   **Compute Resources:** AMD, Anyscale, AWS, Crusoe Cloud, Databricks, DeepInfra, Google Cloud, Intel, Lambda Lab, Nebius, Novita AI, NVIDIA, Replicate, Roblox, RunPod, Trainy, UC Berkeley, UC San Diego
*   **Slack Sponsor:** Anyscale

We also have an official fundraising venue through [OpenCollective](https://opencollective.com/vllm). We plan to use the fund to support the development, maintenance, and adoption of vLLM.

## Citation

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

*   **GitHub Issues:** For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   **vLLM Forum:** For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
*   **Slack:** For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
*   **Security Advisories:** For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
*   **Collaborations & Partnerships:** Contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

## Media Kit

Access vLLM's logo and other media assets in the [media kit repo](https://github.com/vllm-project/media-kit).