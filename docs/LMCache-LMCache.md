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

---

## LMCache: Accelerate LLM Serving with Efficient KV Cache Management

LMCache significantly boosts the performance of Large Language Models (LLMs) by optimizing KV cache utilization, reducing latency, and increasing throughput.  [Explore the LMCache GitHub repository](https://github.com/LMCache/LMCache) for the latest updates and contributions.

**Key Features:**

*   **Reduced TTFT & Increased Throughput:** Drastically improve LLM serving performance, especially in long-context scenarios.
*   **KV Cache Reuse:** Efficiently reuses KV caches of any reused text, not just prefixes, across various serving engine instances.
*   **Integration with vLLM:** Optimized integration with vLLM v1 for enhanced performance, including CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Production-Ready:**  Officially supported in the [vLLM production stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve/kserve).
*   **Non-Prefix Cache Support:** Stable support for non-prefix KV caches enables versatile use cases.
*   **Multiple Storage Options:** Supports CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl) storage for KV caches.
*   **Easy Installation:**  Simple pip installation for Linux NVIDIA GPU platforms.

## Installation

Install LMCache with pip:

```bash
pip install lmcache
```

Detailed installation instructions are available in the [documentation](https://docs.lmcache.ai/getting_started/installation).

## Getting Started

Get started quickly with the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the documentation.

## Documentation

Comprehensive documentation is available online: [LMCache Documentation](https://docs.lmcache.ai/)

## Examples

Explore practical applications with our [examples](https://github.com/LMCache/LMCache/tree/dev/examples).

## Stay Connected

*   **Blog:** [LMCache Blog](https://blog.lmcache.ai/)
*   **Slack:** [Join LMCache Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q)
*   **Interest Form:** [Interest Form](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   **Website:** [LMCache Website](https://lmcache.ai/)
*   **Contact:** [contact@lmcache.ai](mailto:contact@lmcache.ai)
*   **Newsletter:** [Newsletter Signup](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter)

## Community

*   **Roadmap:** [Roadmap](https://github.com/LMCache/LMCache/issues/574)
*   **Community Meeting:** Bi-weekly community meetings on Tuesdays at 9:00 AM PT.  
    *   [Meeting Link](https://uchicago.zoom.us/j/6603596916?pwd=Z1E5MDRWUSt2am5XbEt4dTFkNGx6QT09)
    *   [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
    *   [Meeting Notes](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow)
    *   [YouTube Channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)

## Contributing

We welcome contributions!  See the [Contributing Guide](CONTRIBUTING.md) for details.

## Citation

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

## Social Media

*   [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   [Twitter](https://x.com/lmcache)
*   [YouTube](https://www.youtube.com/@LMCacheTeam)

## License

Licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file.