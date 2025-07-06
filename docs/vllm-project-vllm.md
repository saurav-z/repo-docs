<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

## vLLM: Accelerate Your LLM Inference with Speed and Efficiency

[vLLM](https://github.com/vllm-project/vllm) is a powerful and easy-to-use library designed to make Large Language Model (LLM) serving faster, more efficient, and more accessible to everyone.

**Key Features:**

*   ⚡ **Blazing Fast Performance:**
    *   State-of-the-art serving throughput
    *   **PagedAttention** for efficient memory management
    *   Continuous batching of incoming requests
    *   CUDA/HIP graph for fast model execution
    *   Optimized CUDA kernels and integration with FlashAttention and FlashInfer
    *   Speculative decoding
    *   Chunked prefill
*   ⚙️ **Flexible and Easy to Use:**
    *   Seamless integration with Hugging Face models
    *   High-throughput serving with parallel sampling, beam search, and more
    *   Tensor and pipeline parallelism support
    *   Streaming outputs
    *   OpenAI-compatible API server
    *   Support for NVIDIA, AMD, Intel, and more
    *   Prefix caching and Multi-LoRA support
*   ✅ **Wide Model Compatibility:**
    *   Supports popular open-source models: Llama, Mixtral, Deepseek-V2 and V3, E5-Mistral, LLaVA, and many more (See the full list [here](https://docs.vllm.ai/en/latest/models/supported_models.html)).
*   💡 **Quantization Support:**
    *   GPTQ, AWQ, AutoRound, INT4, INT8, and FP8
---
## About

vLLM, originally from UC Berkeley's Sky Computing Lab, is a community-driven project offering a cutting-edge solution for LLM inference and serving.  Experience significant performance improvements and cost savings when deploying your LLMs.

## Getting Started

Get up and running quickly with vLLM:

1.  **Install:**
    ```bash
    pip install vllm
    ```
2.  **Explore the Documentation:**

    *   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
    *   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
    *   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

Your contributions are welcome! Learn how to get involved in the vLLM project in the [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) section.

## Sponsors

vLLM is a community project supported by the following organizations:

**Cash Donations:** a16z, Dropbox, Sequoia Capital, Skywork AI, ZhenFund

**Compute Resources:** AMD, Anyscale, AWS, Crusoe Cloud, Databricks, DeepInfra, Google Cloud, Intel, Lambda Lab, Nebius, Novita AI, NVIDIA, Replicate, Roblox, RunPod, Trainy, UC Berkeley, UC San Diego

**Slack Sponsor:** Anyscale

Consider supporting vLLM's development via [OpenCollective](https://opencollective.com/vllm).

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

-   For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
-   For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
-   For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
-   For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
-   For collaborations and partnerships, please contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

## Media Kit

Find vLLM's logo and other media assets in our [media kit repo](https://github.com/vllm-project/media-kit).