<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h1 align="center">vLLM: Fast and Efficient LLM Serving</h1>

<p align="center">
  <b>Accelerate your Large Language Model (LLM) inference with vLLM, the open-source framework for serving LLMs efficiently and affordably.</b>
</p>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

## Key Features

vLLM offers a suite of features designed for speed, efficiency, and ease of use.

*   🚀 **Blazing-Fast Inference:** State-of-the-art serving throughput with optimized CUDA kernels, including FlashAttention and FlashInfer, and continuous batching of incoming requests.
*   🧠 **Efficient Memory Management:**  PagedAttention for efficient management of attention key and value memory.
*   🛠️ **Model Quantization Support:** Optimized for GPTQ, AWQ, AutoRound, INT4, INT8, and FP8 for reduced memory footprint and improved performance.
*   🤝 **Broad Model Compatibility:**  Seamless integration with popular Hugging Face models, including Transformer-like LLMs, Mixture-of-Experts LLMs, Embedding Models, and Multi-modal LLMs.  [See supported models](https://docs.vllm.ai/en/latest/models/supported_models.html).
*   ⚙️ **Flexible Serving Options:** High-throughput serving with various decoding algorithms such as parallel sampling, beam search, and more.
*   💻 **Distributed Inference:** Support for tensor, pipeline, data and expert parallelism.
*   🌐 **OpenAI-Compatible API:**  Easy integration with existing infrastructure.
*   🔄 **Streaming and Advanced Features:**  Streaming outputs, speculative decoding, and chunked prefill for enhanced user experience.
*   ⚙️ **Hardware Support:** Runs on NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPUs, and AWS Neuron.
*   💾 **Prefix Caching and Multi-LoRA:** Supports prefix caching and Multi-LoRA for efficient inference with specific use cases.

## About vLLM

vLLM, originally developed at UC Berkeley's Sky Computing Lab, is an open-source project focused on providing a fast and user-friendly solution for Large Language Model (LLM) inference and serving. The project is community-driven, attracting contributions from both academic and industry experts.  [Learn more about vLLM](https://github.com/vllm-project/vllm).

## Getting Started

Quickly install vLLM and start serving your LLMs.

**Installation:**

```bash
pip install vllm
```

For more detailed installation instructions, including building from source, visit the [installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html) in the documentation.

**Resources:**

*   [Documentation](https://docs.vllm.ai/en/latest/)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

Contribute to the vLLM project and help improve the landscape of efficient LLM serving.  We welcome your contributions!  Check out our [Contributing Guide](https://docs.vllm.ai/en/latest/contributing/index.html) to get started.

## Sponsors

vLLM is supported by a vibrant community and the following organizations:

<!-- Note: Please sort them in alphabetical order. -->
<!-- Note: Please keep these consistent with docs/community/sponsors.md -->
**Cash Donations:**

- a16z
- Dropbox
- Sequoia Capital
- Skywork AI
- ZhenFund

**Compute Resources:**

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

**Slack Sponsor:** Anyscale

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

For using vLLM's logo and other assets, please refer to the [media kit repo](https://github.com/vllm-project/media-kit).