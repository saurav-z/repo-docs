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

# LMCache: Accelerate LLM Serving with Efficient KV Cache Management

LMCache dramatically reduces latency and improves throughput for Large Language Models (LLMs) by intelligently caching and reusing KV caches.

**[Explore the LMCache GitHub Repository](https://github.com/LMCache/LMCache)**

## Key Features

*   **Reduced Latency & Increased Throughput:** LMCache significantly speeds up LLM inference, especially in long-context scenarios, by reusing cached KV caches.
*   **Flexible Cache Storage:** Utilize various storage options, including GPU, CPU DRAM, and local disk, for optimal performance and cost efficiency.
*   **vLLM Integration:** Seamlessly integrates with vLLM, enhancing its capabilities with features like high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Non-Prefix KV Cache Support:** LMCache supports the reuse of KV caches for *any* reused text, not just prefixes, providing more comprehensive acceleration.
*   **Production-Ready:** Officially supported in the vLLM [Production Stack](https://github.com/vllm-project/production-stack), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve/kserve).
*   **Multiple Storage Backends:** Supports storage on CPU, Disk and [NIXL](https://github.com/ai-dynamo/nixl).

## Benefits

*   **3-10x Delay Savings:** Experience significant performance improvements in LLM use cases like multi-round QA and RAG when combined with vLLM.
*   **GPU Cycle Reduction:** Conserve valuable GPU resources by reusing cached KV caches.

## Installation

Get started quickly by installing LMCache using pip:

```bash
pip install lmcache
```

For detailed installation instructions, including troubleshooting for specific serving engines and dependencies, consult the [documentation](https://docs.lmcache.ai/getting_started/installation).

## Getting Started

Explore the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the documentation to see LMCache in action.

## Documentation

Comprehensive documentation is available at [https://docs.lmcache.ai/](https://docs.lmcache.ai/) for detailed information, guides, and troubleshooting.

## Examples

Find practical use cases and demonstrations in the [examples](https://github.com/LMCache/LMCache/tree/dev/examples) directory.

## Stay Connected

*   **Website:** [https://lmcache.ai/](https://lmcache.ai/)
*   **Blog:** [https://blog.lmcache.ai/](https://blog.lmcache.ai/)
*   **Slack:** [Join Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3bgx768yd-H8WkOTmPtbxVYJ5nuZ4dmA)
*   **Interest Form:** [Interest Form](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   **Newsletter:** [Sign Up](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter)
*   **Email:** [contact@lmcache.ai](mailto:contact@lmcache.ai)

## Community

*   **Community Meetings:** Bi-weekly on Tuesdays at 9:00 AM PT - [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download). Meeting notes and recordings are available.
*   **Meeting Notes:** [https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow)
*   **YouTube:** [LMCache YouTube Channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)

## Contributing

We welcome contributions! Please refer to the [Contributing Guide](CONTRIBUTING.md) and the [[Onboarding] Welcoming contributors with good first issues!](https://github.com/LMCache/LMCache/issues/627) for details.

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

*   **LinkedIn:** [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   **Twitter:** [Twitter](https://x.com/lmcache)
*   **YouTube:** [Youtube](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.