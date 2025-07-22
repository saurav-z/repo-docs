<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

# vLLM: Serving Large Language Models Made Easy, Fast, and Cheap

**vLLM** revolutionizes LLM inference and serving, providing a high-throughput and user-friendly solution for everyone. ([See the original repo](https://github.com/vllm-project/vllm))

## Key Features

*   **Blazing Fast Performance:**
    *   State-of-the-art serving throughput.
    *   **PagedAttention** for efficient memory management.
    *   Continuous batching of incoming requests.
    *   Fast model execution with CUDA/HIP graphs.
    *   Optimized CUDA kernels, including integration with FlashAttention and FlashInfer.
    *   Speculative decoding and chunked prefill for speed.
*   **Broad Model and Hardware Support:**
    *   Seamless integration with popular Hugging Face models.
    *   Supports Transformer-like, Mixture-of-Expert, and Multi-modal LLMs, including Llama, Mixtral, Deepseek-V2, and LLaVA.
    *   Support for NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
*   **Flexible and User-Friendly:**
    *   High-throughput serving with various decoding algorithms (parallel sampling, beam search, etc.).
    *   Tensor, pipeline, data, and expert parallelism.
    *   Streaming outputs for a responsive user experience.
    *   OpenAI-compatible API server for easy integration.
    *   Prefix caching support.
    *   Multi-LoRA support.
*   **Efficient Quantization:**
    *   Supports GPTQ, AWQ, AutoRound, INT4, INT8, and FP8 quantization for optimized performance and memory usage.

## Getting Started

Install vLLM with `pip`:

```bash
pip install vllm
```

Find more detailed information in the [vLLM Documentation](https://docs.vllm.ai).

## Resources

*   [Documentation](https://docs.vllm.ai)
*   [Blog](https://blog.vllm.ai/)
*   [Paper](https://arxiv.org/abs/2309.06180)
*   [Twitter/X](https://x.com/vllm_project)
*   [User Forum](https://discuss.vllm.ai)
*   [Developer Slack](https://slack.vllm.ai)

## Contributing

We welcome contributions!  Learn how to get involved at [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html).

## Sponsors

vLLM is a community project.  Thank you to our sponsors for providing compute resources and financial support.  (Sorted alphabetically)

**Cash Donations:** a16z, Dropbox, Sequoia Capital, Skywork AI, ZhenFund

**Compute Resources:** AMD, Anyscale, AWS, Crusoe Cloud, Databricks, DeepInfra, Google Cloud, Intel, Lambda Lab, Nebius, Novita AI, NVIDIA, Replicate, Roblox, RunPod, Trainy, UC Berkeley, UC San Diego

**Slack Sponsor:** Anyscale

We also have an official fundraising venue through [OpenCollective](https://opencollective.com/vllm).

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

*   **Technical Questions and Feature Requests:** GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   **Discuss with Fellow Users:** [vLLM Forum](https://discuss.vllm.ai)
*   **Coordinating Contributions & Development:** [Slack](https://slack.vllm.ai)
*   **Security Disclosures:** GitHub [Security Advisories](https://github.com/vllm-project/vllm/security/advisories)
*   **Collaborations and Partnerships:** [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

## Media Kit

Find vLLM's logo and other media assets in the [media kit repo](https://github.com/vllm-project/media-kit).