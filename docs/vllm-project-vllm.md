<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

## vLLM: Unleash the Power of LLMs with Speed, Efficiency, and Ease

[vLLM](https://github.com/vllm-project/vllm) is a pioneering open-source library designed to make Large Language Model (LLM) inference and serving fast, accessible, and cost-effective for everyone. Built with cutting-edge techniques, vLLM empowers you to deploy and utilize LLMs with unparalleled performance.

### Key Features:

*   **Blazing Fast Inference:**
    *   State-of-the-art serving throughput for rapid responses.
    *   **PagedAttention** for efficient memory management, optimized for LLM workloads.
    *   Continuous batching to maximize hardware utilization.
    *   CUDA/HIP graph for fast model execution.
    *   Integration with FlashAttention and FlashInfer for accelerated processing.
    *   Speculative decoding for improved efficiency.
    *   Chunked prefill for faster initial processing.

*   **Extensive Model and Hardware Support:**
    *   Seamlessly integrates with popular Hugging Face models.
    *   Supports a wide range of decoding algorithms, including parallel sampling and beam search.
    *   Supports NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
    *   Comprehensive quantization support: GPTQ, AWQ, AutoRound, INT4, INT8, and FP8.
    *   Prefix caching for streamlined processing.
    *   Multi-LoRA support.

*   **Flexible and Easy to Use:**
    *   OpenAI-compatible API server for straightforward integration.
    *   Supports streaming outputs.
    *   Tensor, pipeline, data, and expert parallelism for distributed inference.

### Supported Models:

vLLM seamlessly supports many popular open-source models, including:

*   Transformer-like LLMs (e.g., Llama)
*   Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
*   Embedding Models (e.g., E5-Mistral)
*   Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

### Getting Started

Install vLLM using pip:

```bash
pip install vllm
```

Or build it [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source).

Explore the full documentation for detailed information:

*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

### Contributing

Join the vLLM community!  We encourage contributions and welcome collaborations.  Learn how to contribute at [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html).

### Sponsors

vLLM is a community-driven project.  We are grateful for the support of our sponsors:

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

### Citation

If you use vLLM in your research, please cite our paper:

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

### Contact Us

*   For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
*   For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
*   For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
*   For collaborations and partnerships, please contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

### Media Kit

Access vLLM's official logo and branding assets in our [media kit repo](https://github.com/vllm-project/media-kit).