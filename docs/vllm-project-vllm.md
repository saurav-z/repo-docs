<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h2 align="center">vLLM:  The Fastest and Most Cost-Effective LLM Serving Solution</h2>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---
## About vLLM

vLLM is revolutionizing Large Language Model (LLM) serving, providing a blazing-fast and easy-to-use platform.  [Visit the original repo](https://github.com/vllm-project/vllm) for the source code.

**Key Features:**

*   **Blazing Fast Performance:**
    *   State-of-the-art serving throughput
    *   Efficient memory management with PagedAttention for optimal resource utilization.
    *   Continuous batching of incoming requests.
    *   Fast model execution with CUDA/HIP graph.
    *   Optimized CUDA kernels, including integration with FlashAttention and FlashInfer.
    *   Support for speculative decoding and chunked prefill.
*   **Wide Range of Quantization Support:**  GPTQ, AWQ, AutoRound, INT4, INT8, and FP8.
*   **Flexible and Easy to Use:**
    *   Seamless integration with popular Hugging Face models.
    *   High-throughput serving with various decoding algorithms, including parallel sampling, beam search, and more.
    *   Tensor and pipeline parallelism support for distributed inference.
    *   Streaming outputs for real-time responses.
    *   OpenAI-compatible API server for easy integration.
    *   Support for NVIDIA, AMD, Intel, and other hardware, including TPU and AWS Neuron.
    *   Prefix caching and Multi-LoRA support.

**Performance:**  vLLM consistently outperforms other LLM serving engines (TensorRT-LLM, SGLang, LMDeploy) in benchmarks.  See the [performance benchmark](https://blog.vllm.ai/2024/09/05/perf-update.html) for detailed results and instructions to reproduce the benchmarks.

**Supported Models:**

vLLM seamlessly supports the most popular open-source models on HuggingFace, including:

*   Transformer-like LLMs (e.g., Llama)
*   Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
*   Embedding Models (e.g., E5-Mistral)
*   Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

---
## Getting Started

Get up and running with vLLM quickly:

```bash
pip install vllm
```

Explore the documentation for detailed information:

*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

---

## Contributing

Contribute to the vLLM project!  See [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for guidelines.

---

## Sponsors

vLLM is a community project supported by these organizations:

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

---

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

---

## Contact Us

-   For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
-   For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
-   For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
-   For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
-   For collaborations and partnerships, please contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

---

## Media Kit

For vLLM logo usage, refer to the [media kit repo](https://github.com/vllm-project/media-kit).