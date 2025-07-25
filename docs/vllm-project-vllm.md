<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

# vLLM: Fast and Easy LLM Serving

**vLLM empowers everyone with easy, fast, and cost-effective Large Language Model (LLM) inference and serving.**  Dive into the future of LLMs with this powerful and efficient open-source library, and explore the original repo [here](https://github.com/vllm-project/vllm).

## Key Features

*   **Blazing Fast Inference:**
    *   State-of-the-art serving throughput for optimized performance.
    *   **PagedAttention** for efficient memory management.
    *   Continuous batching of incoming requests.
    *   Fast model execution using CUDA/HIP graphs.
    *   Optimized CUDA kernels with integration with FlashAttention and FlashInfer.
    *   Speculative decoding & chunked prefill.
*   **Comprehensive Quantization Support:**
    *   Supports various quantization methods including GPTQ, AWQ, AutoRound, INT4, INT8, and FP8.
*   **Flexible and User-Friendly:**
    *   Seamless integration with popular Hugging Face models.
    *   High-throughput serving with various decoding algorithms (parallel sampling, beam search, and more).
    *   Distributed inference with tensor, pipeline, data, and expert parallelism.
    *   Streaming output for real-time results.
    *   OpenAI-compatible API server.
    *   Supports NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
    *   Prefix caching and Multi-LoRA support.
*   **Extensive Model Compatibility:**
    *   Supports a wide range of models, including Transformer-like LLMs (e.g., Llama), Mixture-of-Experts LLMs (e.g., Mixtral), Embedding Models (e.g., E5-Mistral), and Multi-modal LLMs (e.g., LLaVA).

## Getting Started

Install vLLM using pip:

```bash
pip install vllm
```

Explore the documentation to start using vLLM:
*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

Contribute to vLLM's ongoing development by reviewing the [Contributing Guide](https://docs.vllm.ai/en/latest/contributing/index.html).

## Sponsors

vLLM thrives as a community project, supported by numerous organizations providing essential resources for development and testing.

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

**Slack Sponsor:** Anyscale

Support vLLM's continued growth through our [OpenCollective](https://opencollective.com/vllm).

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

*   **Technical Questions and Feature Requests:** GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   **Discussions with Fellow Users:** [vLLM Forum](https://discuss.vllm.ai)
*   **Coordinating Contributions:** [Slack](https://slack.vllm.ai)
*   **Security Disclosures:** GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories)
*   **Collaborations and Partnerships:** [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

## Media Kit

For the official vLLM logo, find the media kit at [vLLM Media Kit](https://github.com/vllm-project/media-kit).