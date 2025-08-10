# LMCache: Accelerate LLM Serving with Efficient Caching

**LMCache drastically improves the performance of your Large Language Model (LLM) applications by caching reusable KV caches, reducing latency and increasing throughput.** Learn more at the [original repository](https://github.com/LMCache/LMCache).

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

## Key Features of LMCache

*   **Reduce TTFT & Boost Throughput:** Significantly decreases Time-To-First-Token and enhances overall throughput, especially in long-context LLM applications.
*   **KV Cache Reuse:** Caches and reuses KV caches of reused text across various locations (GPU, CPU DRAM, Local Disk) to save GPU cycles and reduce user response delay.
*   **vLLM Integration:** Seamless integration with vLLM, including high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Non-Prefix Cache Support:** Stable support for caching non-prefix KV caches, expanding caching capabilities.
*   **Flexible Storage Options:** Supports multiple storage solutions including CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl).
*   **Production Ready:** Officially supported in the [vLLM production stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve/kserve).

## Installation

Install LMCache using pip:

```bash
pip install lmcache
```

*Works on Linux NVIDIA GPU platform.*

For detailed installation instructions, please refer to the [LMCache documentation](https://docs.lmcache.ai/getting_started/installation).

## Getting Started

Explore our [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) to quickly understand and implement LMCache in your projects.

## Documentation

Comprehensive documentation is available at [docs.lmcache.ai](https://docs.lmcache.ai/) to help you get started and dive deeper into the features and functionalities of LMCache.

## Examples

Find practical demonstrations of LMCache in action within our [examples](https://github.com/LMCache/LMCache/tree/dev/examples) directory, showcasing various use cases.

## Stay Connected

*   **Website:** [lmcache.ai](https://lmcache.ai/)
*   **Blog:** [blog.lmcache.ai](https://blog.lmcache.ai/)
*   **Slack:** [Join Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q)
*   **Interest Form:** [Interest Form](https://forms.gle/MHwLiYDU6kcW3dLj7)

## Community

*   **Bi-Weekly Meetings:** Join our community meetings every two weeks on Tuesdays at 9:00 AM PT. [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
*   **Meeting Notes:** Access meeting summaries, discussions, and action items in our [meeting document](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow).
*   **Meeting Recordings:** Watch past meeting recordings on the [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA).

## Contribute

We encourage contributions! Check out our [Contributing Guide](CONTRIBUTING.md) to get involved.

## Citations

If you use LMCache in your research, please cite the following:

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

*   **LinkedIn:** [LMCache LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   **Twitter:** [LMCache Twitter](https://x.com/lmcache)
*   **YouTube:** [LMCache YouTube](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.