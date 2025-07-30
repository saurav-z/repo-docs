# vLLM: Fast and Easy LLM Serving

**Supercharge your Large Language Model (LLM) inference with vLLM, a blazing-fast and user-friendly library designed for efficient and cost-effective LLM serving.** ([View on GitHub](https://github.com/vllm-project/vllm))

## Key Features

*   **Blazing-Fast Inference:** Experience state-of-the-art serving throughput with optimized CUDA kernels, including FlashAttention and FlashInfer, and efficient memory management with PagedAttention.
*   **Efficient Memory Management:**  Utilizes PagedAttention to efficiently manage attention key and value memory.
*   **Continuous Batching:**  Dynamically batches incoming requests for optimal resource utilization.
*   **Quantization Support:** Supports a variety of quantization techniques, including GPTQ, AWQ, AutoRound, INT4, INT8, and FP8, for reduced memory footprint and faster inference.
*   **Decoding Algorithms:** Supports various decoding algorithms, including parallel sampling, beam search, and more for high-throughput serving.
*   **Distributed Inference:** Supports tensor, pipeline, data, and expert parallelism for distributed inference.
*   **Seamless Integration:** Works seamlessly with popular Hugging Face models.
*   **OpenAI-Compatible API:**  Provides an OpenAI-compatible API server for easy integration.
*   **Hardware Agnostic:** Supports NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
*   **Streaming Output:**  Offers streaming output for a responsive user experience.
*   **Model Support:**  Supports a wide range of popular open-source models, including Transformer-like LLMs (e.g., Llama), Mixture-of-Expert LLMs (e.g., Mixtral), Embedding Models (e.g., E5-Mistral), and Multi-modal LLMs (e.g., LLaVA).  See the [supported models list](https://docs.vllm.ai/en/latest/models/supported_models.html).
*   **Prefix Caching & Multi-LoRA:** Includes prefix caching and Multi-LoRA support for flexible model management.

## About vLLM

vLLM, developed at UC Berkeley's Sky Computing Lab, is a community-driven project focused on accelerating and simplifying LLM inference. It's designed to be accessible to everyone, offering a fast, easy, and affordable way to serve LLMs.

## Getting Started

Get up and running with vLLM quickly!

**Installation:**

```bash
pip install vllm
```

For detailed instructions and advanced installation options, visit our [documentation](https://docs.vllm.ai/en/latest/).

**Key Resources:**

*   [Installation Guide](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart Guide](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

Join the vLLM community!  We welcome contributions of all kinds. Learn how to get involved in [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html).

## Sponsors

vLLM is supported by a community of generous sponsors providing both financial and computational resources.  We are grateful for their support!

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

Support vLLM development through [OpenCollective](https://opencollective.com/vllm).

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

*   **GitHub Issues & Discussions:** For technical questions and feature requests, use [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions).
*   **vLLM Forum:** Connect with other users on the [vLLM Forum](https://discuss.vllm.ai).
*   **Slack:** Coordinate contributions and development via [Slack](https://slack.vllm.ai).
*   **Security Advisories:** Report security disclosures using GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories).
*   **Collaborations & Partnerships:** Contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu).

## Media Kit

Find logos and other assets in our [media kit repo](https://github.com/vllm-project/media-kit).