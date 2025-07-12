<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h1 align="center">vLLM: Fast and Easy LLM Serving</h1>

<p align="center">
   Accelerate your Large Language Model (LLM) inference and serving with vLLM, the open-source powerhouse.
</p>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

## Key Features

*   **Blazing Fast Inference:** Experience state-of-the-art throughput with vLLM's optimized architecture.
*   **PagedAttention:** Efficiently manage memory for attention key and value with PagedAttention, crucial for high performance.
*   **Continuous Batching:** Maximize GPU utilization by continuously batching incoming requests.
*   **CUDA/HIP Graph:** Utilize fast model execution with CUDA/HIP graph for improved performance.
*   **Quantization Support:**  Benefit from various quantization techniques (GPTQ, AWQ, AutoRound, INT4, INT8, FP8) to reduce model size and accelerate inference.
*   **Optimized Kernels:** Leverage optimized CUDA kernels, including integration with FlashAttention and FlashInfer for rapid processing.
*   **Speculative Decoding:** Accelerate decoding with speculative decoding, a cutting-edge technique.
*   **Chunked Prefill:** Improve prefill performance with chunked prefill.
*   **Seamless Integration:**  Works effortlessly with popular Hugging Face models.
*   **Decoding Algorithms:** Support high-throughput serving with parallel sampling, beam search, and other decoding algorithms.
*   **Distributed Inference:** Support for Tensor, pipeline, data, and expert parallelism for distributed inference.
*   **Streaming Outputs:** Get results in real-time with streaming outputs.
*   **OpenAI-Compatible API Server:** Easily integrate vLLM into your existing infrastructure.
*   **Hardware Support:** Supports NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPUs, and AWS Neuron.
*   **Prefix Caching:** Benefit from prefix caching support to further optimize inference.
*   **Multi-LoRA Support:** Utilize Multi-LoRA support for advanced use cases.

## About vLLM

vLLM, originally developed at UC Berkeley's Sky Computing Lab, is a fast and easy-to-use library for serving Large Language Models (LLMs). It provides a comprehensive solution for efficient LLM inference and serving, optimizing for speed, ease of use, and cost-effectiveness.  Join the community and contribute to the evolution of LLM serving!  [Visit the original repository](https://github.com/vllm-project/vllm).

### Performance

vLLM delivers exceptional performance. For detailed benchmark results comparing vLLM against other LLM serving engines like TensorRT-LLM, SGLang, and LMDeploy, please refer to our [blog post](https://blog.vllm.ai/2024/09/05/perf-update.html).  You can reproduce the benchmark using our one-click runnable script, located in the [.buildkite/nightly-benchmarks/](.buildkite/nightly-benchmarks/) folder.

### Supported Models

vLLM supports a wide array of popular open-source models, including:

*   Transformer-like LLMs (e.g., Llama)
*   Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
*   Embedding Models (e.g., E5-Mistral)
*   Multi-modal LLMs (e.g., LLaVA)

For a complete list of supported models, please see [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with pip:

```bash
pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source).

Explore the documentation:
*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome contributions!  Review [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) to learn how to get involved.

## Sponsors

vLLM is a community-driven project supported by the following organizations.  Thank you!

*   (Alphabetical order)
    *   a16z
    *   AMD
    *   Anyscale
    *   AWS
    *   Crusoe Cloud
    *   Databricks
    *   DeepInfra
    *   Dropbox
    *   Google Cloud
    *   Intel
    *   Lambda Lab
    *   Nebius
    *   Novita AI
    *   NVIDIA
    *   Replicate
    *   Roblox
    *   RunPod
    *   Sequoia Capital
    *   Skywork AI
    *   Trainy
    *   UC Berkeley
    *   UC San Diego
    *   ZhenFund

  *  **Slack Sponsor**: Anyscale

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

*   For technical questions and feature requests, use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions).
*   Discuss with fellow users on the [vLLM Forum](https://discuss.vllm.ai).
*   Coordinate contributions and development on [Slack](https://slack.vllm.ai).
*   Report security disclosures via GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories).
*   For collaborations and partnerships, contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu).

## Media Kit

Access vLLM's logo and other media assets in our [media kit repo](https://github.com/vllm-project/media-kit).