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

LMCache dramatically reduces latency and boosts throughput for Large Language Models by intelligently caching reusable text, achieving significant speedups.  [Check out the original repo](https://github.com/LMCache/LMCache)!

## Key Features

*   **Performance Boost:** Significantly reduces Time-To-First-Token (TTFT) and increases throughput for LLMs, especially in long-context scenarios.
*   **KV Cache Reuse:** Efficiently reuses KV caches of *any* reused text, not just prefixes, across different serving engine instances.
*   **Integration with vLLM:** Optimized integration with vLLM v1, offering features like:
    *   High-performance CPU KVCache offloading
    *   Disaggregated prefill
    *   P2P KVCache sharing
*   **Broad Compatibility:** Supported by the vLLM production stack, llm-d, and KServe.
*   **Flexible Storage Options:** Supports CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl) storage for KV caches.
*   **Non-Prefix KV Cache Support:** Stable support for caching non-prefix KV caches.

## What is LMCache?

LMCache is an LLM serving engine extension designed to optimize performance by caching the KV caches of reusable texts. This results in reduced TTFT and higher throughput, particularly in applications dealing with long contexts, such as multi-round QA and RAG. By storing KV caches across various locations (GPU, CPU DRAM, Local Disk), LMCache enables the reuse of cached text, leading to significant GPU cycle savings and improved user response times.

## Benefits of Using LMCache

*   **Reduced Latency:** Experience 3-10x delay savings in many LLM use cases when combined with vLLM.
*   **GPU Resource Optimization:** Reduce the demand on precious GPU cycles.
*   **Enhanced User Experience:** Faster response times for improved user satisfaction.

## Installation

Install LMCache using pip:

```bash
pip install lmcache
```

Works on Linux NVIDIA GPU platform.

For detailed installation instructions, see the [installation documentation](https://docs.lmcache.ai/getting_started/installation).

## Getting Started

Explore the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in our documentation to get started quickly.

## Resources

*   **Documentation:** Comprehensive [documentation](https://docs.lmcache.ai/) for detailed information.
*   **Blog:** Stay updated with the latest news and insights on the [LMCache blog](https://blog.lmcache.ai/).
*   **Examples:** Explore practical use cases in our [examples](https://github.com/LMCache/LMCache/tree/dev/examples) directory.
*   **Community:**
    *   Join the [LMCache Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q)
    *   Attend our [weekly community meeting](https://uchicago.zoom.us/j/6603596916?pwd=Z1E5MDRWUSt2am5XbEt4dTFkNGx6QT09) (alternating between Tuesdays at 9:00 AM PT and 6:30 PM PT - see links in original README).
    *   View meeting notes and recordings on the [community document](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow) and [YouTube channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA), respectively.
    *   Stay informed by filling out the [interest form](https://forms.gle/MHwLiYDU6kcW3dLj7) or signing up for the [newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter).
    *   Contact us at contact@lmcache.ai

## Contributing

Contributions are welcome! See our [Contributing Guide](CONTRIBUTING.md) for details.

## Citation

If you use LMCache in your research, please cite our papers:

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