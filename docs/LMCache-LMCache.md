<div align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/LMCache/LMCache/dev/asset/logo.png" width="720" alt="lmcache logo">
  </p>
</div>

[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://docs.lmcache.ai/)
[![PyPI](https://img.shields.io/pypi/v/lmcache)](https://pypi.org/project/lmcache/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lmcache)](https://pypi.org/project/lmcache/)
[![Unit Tests](https://badge.buildkite.com/ce25f1819a274b7966273bfa54f0e02f092c3de0d7563c5c9d.svg)](https://buildkite.com/lmcache/lmcache-unittests)
[![Code Quality](https://github.com/lmcache/lmcache/actions/workflows/code_quality_checks.yml/badge.svg?branch=dev&label=tests)](https://github.com/LMCache/LMCache/actions/workflows/code_quality_checks.yml)
[![Integration Tests](https://badge.buildkite.com/108ddd4ab482a2480999dec8c62a640a3315ed4e6c4e86798e.svg)](https://buildkite.com/lmcache/lmcache-vllm-integration-tests)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10841/badge)](https://www.bestpractices.dev/projects/10841)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/LMCache/LMCache/badge)](https://scorecard.dev/viewer/?uri=github.com/LMCache/LMCache)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/LMCache/LMCache/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/w/LMCache/LMCache)](https://github.com/LMCache/LMCache/graphs/commit-activity)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/lmcache)](https://pypi.org/project/lmcache/)
[![YouTube Channel Views](https://img.shields.io/youtube/channel/views/UC58zMz55n70rtf1Ak2PULJA)](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)

[**View the LMCache Repository on GitHub**](https://github.com/LMCache/LMCache)

---

## LMCache: Accelerate LLM Serving with Intelligent KV Cache Management

LMCache is an innovative LLM serving engine extension designed to drastically reduce Time-To-First-Token (TTFT) and boost throughput.

### Key Features

*   **Enhanced Performance:** Drastically reduces TTFT and improves LLM throughput, especially in long-context scenarios.
*   **KV Cache Reuse:** Efficiently reuses KV caches for repeated text across different instances, saving GPU cycles.
*   **vLLM Integration:** Seamlessly integrates with vLLM, offering high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Production-Ready:** Officially supported in the vLLM production stack, llm-d, and KServe.
*   **Flexible Storage:** Supports KV cache storage on CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl) for optimized resource utilization.
*   **Non-Prefix Cache Support:** Stable support for non-prefix KV caches, enhancing reusability.

### Installation

Get started with LMCache using pip:

```bash
pip install lmcache
```

Detailed installation instructions are available in the [documentation](https://docs.lmcache.ai/getting_started/installation).

### Getting Started

Explore the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the documentation to learn how to integrate LMCache into your projects.

### Documentation

Comprehensive documentation is available [here](https://docs.lmcache.ai/).

### Examples

Explore practical [examples](https://github.com/LMCache/LMCache/tree/dev/examples) to see LMCache in action.

### Stay Connected

*   **Blog:** [LMCache Blog](https://blog.lmcache.ai/)
*   **Join the Community:** [Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q)
*   **Newsletter:** [Sign up for our Newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter)
*   **Website:** [LMCache Website](https://lmcache.ai/)
*   **Contact:** [Email](contact@lmcache.ai)
*   **Community Meeting:** [Community Meeting]( https://uchicago.zoom.us/j/6603596916?pwd=Z1E5MDRWUSt2am5XbEt4dTFkNGx6QT09)
*   **LinkedIn:** [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   **Twitter:** [Twitter](https://x.com/lmcache)
*   **Youtube:** [Youtube](https://www.youtube.com/@LMCacheTeam)

### Contribute

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Citation

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

### License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.