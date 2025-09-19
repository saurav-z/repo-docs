# LMCache: Accelerate LLM Serving with Efficient KV Cache Management

**LMCache drastically reduces latency and GPU costs for Large Language Models by intelligently caching and reusing KV caches, leading to significant performance gains.** ([Original Repository](https://github.com/LMCache/LMCache))

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

<br />

**Explore how LMCache can revolutionize your LLM deployments**: [Blog](https://blog.lmcache.ai/) | [Documentation](https://docs.lmcache.ai/) | [Join Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3bgx768yd-H8WkOTmPtbxVYJ5nuZ4dmA) | [Interest Form](https://forms.gle/MHwLiYDU6kcW3dLj7) | [Roadmap](https://github.com/LMCache/LMCache/issues/1253)

🔥 **Enterprise-Ready**:  LMCache is production-ready and officially supported in the [vLLM Production Stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve/).

## Key Features

*   **Accelerated LLM Performance:** Drastically reduces Time-To-First-Token (TTFT) and increases throughput, especially in long-context scenarios.
*   **KV Cache Reuse:** Caches and reuses KV caches of any reused text (not necessarily prefix) across serving engine instances, saving GPU cycles.
*   **Seamless Integration:** Integrates with vLLM v1 with high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Flexible Storage Options:** Supports KV cache storage on CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl).
*   **Stable Non-Prefix Support:**  Reliable support for caching and reusing non-prefix KV caches.
*   **Easy to Deploy**: Support in the [vLLM production stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve/kserve) .

## Performance Benefits

LMCache, combined with vLLM, delivers substantial performance improvements, including 3-10x delay savings and significant GPU cycle reduction in applications like multi-round QA and RAG.

![performance](https://github.com/user-attachments/assets/86137f17-f216-41a0-96a7-e537764f7a4c)

## Installation

Install LMCache easily via pip:

```bash
pip install lmcache
```

Requires a Linux NVIDIA GPU platform.  Consult the [documentation](https://docs.lmcache.ai/getting_started/installation) for detailed installation instructions, particularly if using a different serving engine or encountering "undefined symbol" or torch mismatch issues.

## Getting Started

Get started with LMCache quickly using our [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the docs.

## Documentation

Comprehensive [documentation](https://docs.lmcache.ai/) is available online to guide you.  Keep up with the latest developments on the [LMCache blog](https://blog.lmcache.ai/).

## Examples

Explore practical use cases with our [examples](https://github.com/LMCache/LMCache/tree/dev/examples).

## Community and Support

*   **Stay Connected**: Fill out the [interest form](https://forms.gle/mQfQDUXbKfp2St1z7), [sign up for our newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter), [join LMCache slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ), [check out LMCache website](https://lmcache.ai/), or [drop an email](mailto:contact@lmcache.ai).
*   **Community Meetings**: Join bi-weekly community meetings on Tuesdays at 9:00 AM PT - [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download). Meeting notes are available [here](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow). Recordings are on the [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA).

## Contributing

We welcome contributions!  See the [Contributing Guide](CONTRIBUTING.md). Check out [[Onboarding] Welcoming contributors with good first issues!](https://github.com/LMCache/LMCache/issues/627).

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

## Social Media

*   [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   [Twitter](https://x.com/lmcache)
*   [YouTube](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.