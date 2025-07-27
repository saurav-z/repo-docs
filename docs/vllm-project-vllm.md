<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

# vLLM: Serving Large Language Models (LLMs) Made Easy, Fast, and Cheap

vLLM empowers you to serve LLMs with unparalleled speed and efficiency. Find the original repo [here](https://github.com/vllm-project/vllm).

## Key Features

*   **Blazing Fast Inference:** Experience state-of-the-art throughput and low latency.
*   **Efficient Memory Management:**  Leverage PagedAttention for optimized memory use.
*   **Continuous Batching:**  Maximize GPU utilization with dynamic batching of requests.
*   **Model Compatibility:**  Seamlessly supports a wide range of Hugging Face models.
*   **Quantization Support:** Offers various quantization techniques (GPTQ, AWQ, AutoRound, INT4, INT8, FP8).
*   **Distributed Inference:** Supports tensor, pipeline, data, and expert parallelism.
*   **OpenAI-Compatible API:** Easy integration with existing infrastructure.
*   **Versatile Hardware Support:** Runs on NVIDIA, AMD, Intel, PowerPC, TPU, and AWS Neuron.
*   **Multi-LoRA Support:** Enhanced flexibility.
*   **Prefix Caching:** Improved performance.

## About vLLM

vLLM is a cutting-edge library designed to make LLM inference and serving accessible and efficient. Developed by the Sky Computing Lab at UC Berkeley and now a thriving community project, vLLM focuses on speed, ease of use, and cost-effectiveness.  It incorporates innovative techniques like PagedAttention to optimize memory usage, enabling high-throughput serving. vLLM's flexibility allows it to support a broad spectrum of open-source models, making it an ideal solution for various AI applications.

## Getting Started

Install vLLM using pip:

```bash
pip install vllm
```

Explore the [documentation](https://docs.vllm.ai/en/latest/) for:

*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome contributions from everyone!  Check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) to learn how you can get involved.

## Sponsors

vLLM thrives thanks to community support.  We are grateful for our sponsors:

*   **Cash Donations:** a16z, Dropbox, Sequoia Capital, Skywork AI, ZhenFund
*   **Compute Resources:** AMD, Anyscale, AWS, Crusoe Cloud, Databricks, DeepInfra, Google Cloud, Intel, Lambda Lab, Nebius, Novita AI, NVIDIA, Replicate, Roblox, RunPod, Trainy, UC Berkeley, UC San Diego
*   **Slack Sponsor:** Anyscale

Support vLLM's development via [OpenCollective](https://opencollective.com/vllm).

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

*   **Technical Questions and Feature Requests:** [GitHub Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   **Community Discussions:** [vLLM Forum](https://discuss.vllm.ai)
*   **Development Coordination:** [Slack](https://slack.vllm.ai)
*   **Security Disclosures:** GitHub [Security Advisories](https://github.com/vllm-project/vllm/security/advisories)
*   **Collaborations and Partnerships:** [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

## Media Kit

Access the vLLM logo and other media assets from [our media kit repo](https://github.com/vllm-project/media-kit).