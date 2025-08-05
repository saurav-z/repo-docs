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

## LMCache: Accelerate LLM Inference with Efficient KV Cache Management

LMCache is a powerful extension designed to supercharge Large Language Model (LLM) serving, reducing Time-To-First-Token (TTFT) and boosting throughput. Explore the [LMCache GitHub Repository](https://github.com/LMCache/LMCache) for the latest updates.

**Key Features:**

*   **TTFT Reduction & Throughput Increase:** LMCache optimizes LLM serving by caching reusable KV caches, leading to faster response times and higher throughput, especially in long-context scenarios.
*   **Reusable KV Caches:**  Efficiently reuses KV caches for repeated text across any instance, saving precious GPU cycles.
*   **vLLM Integration:**  Seamlessly integrates with vLLM, offering high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Production Ready:** Supported by vLLM production stack, llm-d and KServe.
*   **Non-Prefix KV Cache Support:** Offers stable support for non-prefix KV caches.
*   **Versatile Storage Options:** Supports storage on CPU, disk, and [NIXL](https://github.com/ai-dynamo/nixl).
*   **Easy Installation:** Install with pip and compatible with Linux NVIDIA GPU platforms.

## Key Benefits

*   **Reduced Latency:** Experience 3-10x delay savings in many LLM use cases by combining LMCache with vLLM.
*   **Optimized GPU Utilization:** Reduce GPU cycle consumption for more efficient resource management.
*   **Improved User Experience:** Faster response times lead to a more responsive and engaging user experience.

## Installation

Get started with LMCache in a few simple steps:

```bash
pip install lmcache
```

For detailed instructions, refer to the [Installation Guide](https://docs.lmcache.ai/getting_started/installation) in our documentation.

## Getting Started

Dive into LMCache with our [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) to see how it works.

## Documentation

Find comprehensive documentation and learn more about LMCache at [https://docs.lmcache.ai/](https://docs.lmcache.ai/).  Stay updated with our [blog](https://blog.lmcache.ai/) for the latest news and insights.

## Examples

Explore practical applications with our [examples](https://github.com/LMCache/LMCache/tree/dev/examples) demonstrating different use cases.

## Community

*   **Join the Community:** Fill out the [interest form](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   **Stay Updated:** Sign up for our [newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter)
*   **Connect on Slack:** Join the [LMCache Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ)
*   **Visit our Website:** [https://lmcache.ai/](https://lmcache.ai/)
*   **Contact Us:** [contact@lmcache.ai](mailto:contact@lmcache.ai)

## Community Meetings

Join our bi-weekly community meetings:

*   **Schedule:** Tuesdays at 9:00 AM PT - [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
*   **Meeting Notes:** [Meeting Notes](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow)
*   **Meeting Recordings:** [YouTube Channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)

## Contributing

We welcome contributions!  Please review our [Contributing Guide](CONTRIBUTING.md) to get started.

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
*   [YouTube](https://www.youtube.com/@LMCacheTeam)

## License

The LMCache codebase is licensed under Apache License 2.0. See the [LICENSE](LICENSE) file for details.