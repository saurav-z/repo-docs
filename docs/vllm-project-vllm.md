<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

## vLLM: Serve Large Language Models (LLMs) Faster, Easier, and More Affordably

[vLLM](https://github.com/vllm-project/vllm) empowers you to efficiently serve LLMs, making them accessible for everyone.

**Key Features:**

*   **Blazing Fast Inference:**
    *   State-of-the-art serving throughput for optimal performance.
    *   Efficient memory management with **PagedAttention**.
    *   Continuous batching of requests to maximize resource utilization.
    *   Optimized CUDA/HIP graph for rapid model execution.
    *   Support for various quantizations (GPTQ, AWQ, AutoRound, INT4, INT8, FP8).
    *   Utilizes optimized CUDA kernels including FlashAttention and FlashInfer.
    *   Includes speculative decoding and chunked prefill for further acceleration.
*   **User-Friendly and Flexible:**
    *   Seamless integration with popular Hugging Face models.
    *   Supports diverse decoding algorithms (parallel sampling, beam search, etc.).
    *   Offers tensor, pipeline, data, and expert parallelism for distributed inference.
    *   Provides streaming output for real-time results.
    *   Includes an OpenAI-compatible API server for easy integration.
    *   Supports NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
    *   Prefix caching support for faster processing of repeated sequences.
    *   Multi-LoRA support for model customization.
*   **Broad Model Compatibility:**
    *   Supports most popular open-source models from Hugging Face.
    *   Works with Transformer-based LLMs (e.g., Llama).
    *   Compatible with Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3).
    *   Includes support for Embedding Models (e.g., E5-Mistral).
    *   Offers support for Multi-modal LLMs (e.g., LLaVA).

**Getting Started**

Install vLLM using pip:

```bash
pip install vllm
```

Comprehensive documentation is available to guide you:

*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

**Contribute**

Your contributions are highly valued.  Check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for details on how to get involved.

**Sponsors**

vLLM is a community-driven project supported by generous contributions.  Thank you to our sponsors:

*   **Cash Donations:** a16z, Dropbox, Sequoia Capital, Skywork AI, ZhenFund
*   **Compute Resources:** AMD, Anyscale, AWS, Crusoe Cloud, Databricks, DeepInfra, Google Cloud, Intel, Lambda Lab, Nebius, Novita AI, NVIDIA, Replicate, Roblox, RunPod, Trainy, UC Berkeley, UC San Diego
*   **Slack Sponsor:** Anyscale
*   **OpenCollective:** Consider supporting vLLM through [OpenCollective](https://opencollective.com/vllm).

**Citation**

If you use vLLM in your research, please cite our paper:

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

**Contact Us**

*   **Technical Questions & Feature Requests:**  GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   **Community Discussion:** [vLLM Forum](https://discuss.vllm.ai)
*   **Development Coordination:** [Slack](https://slack.vllm.ai)
*   **Security Disclosures:** GitHub [Security Advisories](https://github.com/vllm-project/vllm/security/advisories)
*   **Collaborations & Partnerships:** [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

**Media Kit**

For vLLM logo usage, please refer to [our media kit repo](https://github.com/vllm-project/media-kit).