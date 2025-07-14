<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

## vLLM: Supercharge Your LLM Inference with Speed and Efficiency

vLLM is a fast and easy-to-use library for LLM inference and serving, offering state-of-the-art performance and flexibility.  [Check out the original repository here](https://github.com/vllm-project/vllm).

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

### Key Features

*   **Blazing Fast Inference:** Experience state-of-the-art serving throughput with optimized CUDA kernels, PagedAttention, and continuous batching.
*   **Efficient Memory Management:**  Benefit from PagedAttention, designed for efficient management of attention key and value memory.
*   **Broad Hardware and Model Support:** Works with NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron, plus seamless integration with popular Hugging Face models, including Transformers, Mixture-of-Experts (MoE) models, and more!
*   **Quantization Support:** Utilize various quantization techniques like GPTQ, AWQ, AutoRound, INT4, INT8, and FP8 for faster inference and reduced memory footprint.
*   **Flexible Decoding Algorithms:** Leverage high-throughput serving with parallel sampling, beam search, and other decoding algorithms.
*   **Distributed Inference:** Take advantage of Tensor, pipeline, data, and expert parallelism for distributed inference.
*   **OpenAI-Compatible API Server:**  Easily integrate vLLM into your existing infrastructure.
*   **Continuous Innovation:** Benefit from ongoing development and optimization, including speculative decoding and chunked prefill.

### Performance Benchmarks
*   **Superior Performance:** vLLM's performance against LLM serving engines like TensorRT-LLM, SGLang, and LMDeploy is documented in our blog post.  Reproduce the benchmark easily with our one-click script, available in the nightly-benchmarks folder.

### Getting Started

Install vLLM easily using pip:

```bash
pip install vllm
```

Or, [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source).

Explore our comprehensive [documentation](https://docs.vllm.ai/en/latest/) for:

*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

### Contributing

We encourage contributions from the community!  Learn how to get involved in [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html).

### Sponsors

vLLM is a community-driven project, and we appreciate the support of our sponsors:

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

Consider supporting vLLM's development through our [OpenCollective](https://opencollective.com/vllm).

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

Access the vLLM logo and media assets in our [media kit repo](https://github.com/vllm-project/media-kit).