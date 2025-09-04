<div align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/LMCache/LMCache/dev/asset/logo.png" width="720" alt="lmcache logo">
  </p>

  [![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://docs.lmcache.ai/)
  [![PyPI](https://img.shields.io/pypi/v/lmcache)](https://pypi.org/project/lmcache/)
  [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lmcache)](https://pypi.org/project/lmcache/)
  [![Unit Tests](https://badge.buildkite.com/ce25f1819a274b7966273bfa54f0e02f092c3de0d7563c5c9d.svg)](https://buildkite.com/lmcache/lmcache-unittests)
  [![Code Quality](https://github.com/lmcache/lmcache/actions/workflows/code_quality_checks.yml/badge.svg?branch=dev&label=tests)](https://github.com/LMCache/LMCache/actions/workflows/code_quality_checks.yml)
  [![Integration Tests](https://badge.buildkite.com/108ddd4ab482a2480999dec8c62a640a3315ed4e6c4e86798e.svg)](https://buildkite.com/lmcache/lmcache-vllm-integration-tests)

   <br />

  [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10841/badge)](https://www.bestpractices.dev/projects/10841)
  [![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/LMCache/LMCache/badge)](https://scorecard.dev/viewer/?uri=github.com/LMCache/LMCache)
  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/LMCache/LMCache/)
  [![GitHub commit activity](https://img.shields.io/github/commit-activity/w/LMCache/LMCache)](https://github.com/LMCache/LMCache/graphs/commit-activity)
  [![PyPI - Downloads](https://img.shields.io/pypi/dm/lmcache)](https://pypi.org/project/lmcache/)
  [![YouTube Channel Views](https://img.shields.io/youtube/channel/views/UC58zMz55n70rtf1Ak2PULJA)](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)
</div>

## LMCache: Accelerate LLM Serving with Efficient KV Cache Management

LMCache optimizes Large Language Model (LLM) performance by intelligently caching and reusing KV caches.  [Explore the original repository](https://github.com/LMCache/LMCache) for more details.

**Key Features:**

*   **Reduced TTFT (Time To First Token) & Increased Throughput:** Significantly boosts LLM serving performance, especially for long-context applications.
*   **Efficient KV Cache Management:**  Stores KV caches across GPU, CPU DRAM, and local disk to maximize reuse.
*   **Universal Reuse:** Reuses KV caches for any reused text, regardless of position or instance.
*   **vLLM Integration:**  Seamlessly integrates with vLLM, offering high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Production-Ready:** Supported in vLLM production stack, llm-d, and KServe.
*   **Flexible Storage Options:** Supports CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl) storage for KV caches.
*   **Non-Prefix KV Cache Support:** Provides stable support for caching non-prefix KV caches.

## Installation

Install LMCache easily using pip:

```bash
pip install lmcache
```

For detailed installation instructions and troubleshooting, especially if you're not using the latest vLLM or have dependency conflicts, refer to the [LMCache documentation](https://docs.lmcache.ai/getting_started/installation).

## Getting Started

Quickly dive in with our [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the documentation.

## Documentation

Comprehensive documentation is available at [https://docs.lmcache.ai/](https://docs.lmcache.ai/).

## Examples

Explore practical use cases with our [examples](https://github.com/LMCache/LMCache/tree/dev/examples).

## Community & Support

*   **Join the Community:** [Join Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q) and participate in bi-weekly [community meetings]( https://uchicago.zoom.us/j/6603596916?pwd=Z1E5MDRWUSt2am5XbEt4dTFkNGx6QT09).
*   **Stay Updated:** Subscribe to our [newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter) and follow us on [social media](#socials) and the [LMCache blog](https://blog.lmcache.ai/).
*   **Contact Us:** Reach out to us via email: [contact@lmcache.ai](mailto:contact@lmcache.ai).
*   **Interest Form:** Let us know your interests here: [interest form](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   **Roadmap:** [Roadmap](https://github.com/LMCache/LMCache/issues/1253)

## Contributing

Contributions are welcome! Review the [Contributing Guide](CONTRIBUTING.md) and check out the [good first issues](https://github.com/LMCache/LMCache/issues/627) for getting started.

## Citation

If you use LMCache in your research, please cite the following papers:
```
@inproceedings{liu2024cachegen,
  title={Cachegen: Kv cache compression and streaming for fast large language model serving},
  author={Liu, Yuhan and Li, Hanchen and Cheng, Yihua and Ray, Siddhant and Huang, Yuyang and Zhang, Qizheng and Du, Kuntai and Yao, Jiayi and Lu, Shan and Ananthanarayanan, Ganesh and others},
  booktitle={Proceedings of the ACM SIGCOMM 2024 Conference},
  pages={38--56},
  year={2024}
}

@article{cheng2024large,
  title={Do Large Language Models Need a Content Delivery Network?},
  author={Cheng, Yihua and Du, Kuntai and Yao, Jiayi and Jiang, Junchen},
  journal={arXiv preprint arXiv:2409.13761},
  year={2024}
}

@inproceedings{10.1145/3689031.3696098,
  author = {Yao, Jiayi and Li, Hanchen and Liu, Yuhan and Ray, Siddhant and Cheng, Yihua and Zhang, Qizheng and Du, Kuntai and Lu, Shan and Jiang, Junchen},
  title = {CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion},
  year = {2025},
  url = {https://doi.org/10.1145/3689031.3696098},
  doi = {10.1145/3689031.3696098},
  booktitle = {Proceedings of the Twentieth European Conference on Computer Systems},
  pages = {94–109},
}
```

## Socials

*   [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   [Twitter](https://x.com/lmcache)
*   [YouTube](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
```
Key changes and why:

*   **SEO Optimization:**  The title includes keywords like "LLM serving," "KV cache," and "performance." The introduction summarizes the purpose of the software.
*   **Clear Headings:** Uses clear headings (e.g., Installation, Getting Started) to structure the information and improve readability.
*   **Bulleted Key Features:**  Uses bullet points to highlight the main benefits and features, making them easy to scan.
*   **Concise Summary:** Starts with a strong, one-sentence hook to grab attention.
*   **Call to Action:** Includes links to documentation, examples, and the community to encourage engagement.
*   **Conciseness:** Removes redundant information and keeps the descriptions focused.
*   **Structure:**  Organizes the content logically.
*   **Emphasis:**  Uses bolding to highlight important terms.
*   **Complete Information:** Includes all the important links and information from the original README.