# vLLM: Fast and Easy LLM Serving for Everyone

**vLLM revolutionizes Large Language Model (LLM) serving, making it fast, efficient, and accessible to all.  ([View the Original Repo](https://github.com/vllm-project/vllm))**

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

**Key Features:**

*   **Blazing-Fast Performance:** State-of-the-art serving throughput with PagedAttention for efficient memory management and CUDA/HIP graph for fast model execution.
*   **Efficient Memory Management:** Leverages PagedAttention for optimized key and value memory management.
*   **Continuous Batching:** Processes incoming requests in continuous batches for improved efficiency.
*   **Quantization Support:** Includes support for GPTQ, AWQ, AutoRound, INT4, INT8, and FP8 quantization.
*   **Optimized Kernels:** Integrates with FlashAttention, FlashInfer, and optimized CUDA kernels.
*   **Versatile Decoding Algorithms:** Supports parallel sampling, beam search, and other decoding methods.
*   **Distributed Inference:** Offers tensor, pipeline, data, and expert parallelism for distributed inference.
*   **OpenAI-Compatible API:** Provides an OpenAI-compatible API server for easy integration.
*   **Broad Hardware Compatibility:** Supports NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPUs, and AWS Neuron.
*   **Model Compatibility:** Seamlessly supports most popular open-source models on Hugging Face, including Transformers-like LLMs, Mixture-of-Experts, Embedding Models, and Multi-modal LLMs.
*   **Multi-LoRA support**

## About vLLM

vLLM, initially developed at UC Berkeley's Sky Computing Lab, is a leading open-source library designed for efficient LLM inference and serving. It's a community-driven project that empowers developers to deploy and utilize LLMs with unprecedented speed and ease.

## Getting Started

vLLM is easy to install and use.

**Installation:**

```bash
pip install vllm
```

**Explore Further:**

*   [Documentation](https://docs.vllm.ai): Comprehensive documentation for in-depth understanding.
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html): Get up and running quickly.
*   [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html): Find a list of supported models.

## Latest News

*   [2024/05] vLLM is now a hosted project under PyTorch Foundation! Please find the announcement [here](https://pytorch.org/blog/pytorch-foundation-welcomes-vllm/).
*   [2024/01] We are excited to announce the alpha release of vLLM V1: A major architectural upgrade with 1.7x speedup! Clean code, optimized execution loop, zero-overhead prefix caching, enhanced multimodal support, and more. Please check out our blog post [here](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html).
*   [2024/07] In partnership with Meta, vLLM officially supports Llama 3.1 with FP8 quantization and pipeline parallelism! Please check out our blog post [here](https://blog.vllm.ai/2024/07/23/llama31.html).
*   [and more - see original README for more updates]

<details>
<summary>Previous News</summary>
  [See original README for previous news]
</details>

## Contributing

We welcome contributions from the community! Learn how to contribute [here](https://docs.vllm.ai/en/latest/contributing/index.html).

## Sponsors

vLLM is supported by a vibrant community. We are grateful for the support of these organizations:

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

*Slack Sponsor: Anyscale*

We also have an official fundraising venue through [OpenCollective](https://opencollective.com/vllm). We plan to use the fund to support the development, maintenance, and adoption of vLLM.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

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

For vLLM logo usage, please refer to [our media kit repo](https://github.com/vllm-project/media-kit).