# LMCache: Accelerate LLM Serving with Intelligent KV Cache Management

**LMCache dramatically reduces latency and boosts throughput for Large Language Models by intelligently caching and reusing KV cache data.** ([Back to the original repo](https://github.com/LMCache/LMCache))

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

<br/>

**LMCache is officially supported in [llm-d](https://github.com/llm-d/llm-d/) and [KServe](https://github.com/kserve)!**

## Key Features

*   **Reduced TTFT and Increased Throughput:** LMCache significantly speeds up LLM inference, especially for long-context applications.
*   **KV Cache Reusability:** Stores and reuses KV caches of reused text across GPU, CPU DRAM, and local disk.
*   **vLLM Integration:** Seamlessly integrates with vLLM, with features like high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Non-Prefix Cache Support:** Offers stable support for non-prefix KV caches, enhancing flexibility.
*   **Multiple Storage Options:** Supports CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl) for KV cache storage.
*   **Easy Installation:** Simple `pip install lmcache` installation process.
*   **Production Stack Support:** Ready for enterprise deployment, with support in the [vLLM production stack](https://github.com/vllm-project/production-stack/).

## How LMCache Works

LMCache optimizes LLM performance by caching and reusing KV caches. This reduces the need for repeated computations, leading to faster response times and more efficient use of GPU resources.  By combining LMCache with vLLM, developers can achieve significant performance gains in multi-round QA and RAG.

## Installation

Get started with LMCache quickly using pip:

```bash
pip install lmcache
```

Detailed installation instructions and troubleshooting tips are available in the [documentation](https://docs.lmcache.ai/getting_started/installation).

## Getting Started

Explore the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) to begin using LMCache.

## Documentation

Comprehensive documentation is available at [https://docs.lmcache.ai/](https://docs.lmcache.ai/) and the [blog](https://blog.lmcache.ai/).

## Examples

Explore practical use cases with our comprehensive [examples](https://github.com/LMCache/LMCache/tree/dev/examples).

## Community

*   [Join Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3bgx768yd-H8WkOTmPtbxVYJ5nuZ4dmA)
*   [Interest Form](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   [Roadmap](https://github.com/LMCache/LMCache/issues/1253)
*   [LMCache website](https://lmcache.ai/)
*   [Community meeting]( https://uchicago.zoom.us/j/6603596916?pwd=Z1E5MDRWUSt2am5XbEt4dTFkNGx6QT09) for bi-weekly meeting.
    *   Meetings are held bi-weekly on: Tuesdays at 9:00 AM PT – [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
    *   Meeting notes: [document](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow)
    *   [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA).
*   Contact us at: [contact@lmcache.ai](mailto:contact@lmcache.ai)

## Contributing

We welcome contributions!  See the [Contributing Guide](CONTRIBUTING.md) and [[Onboarding] Welcoming contributors with good first issues!](https://github.com/LMCache/LMCache/issues/627)

## Citation

If you use LMCache for your research, please cite our papers:

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
*   [Youtube](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.