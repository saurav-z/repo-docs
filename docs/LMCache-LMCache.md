<div align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/LMCache/LMCache/dev/asset/logo.png" width="720" alt="LMCache Logo">
  </p>
</div>

[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://docs.lmcache.ai/)
[![PyPI](https://img.shields.io/pypi/v/lmcache)](https://pypi.org/project/lmcache/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lmcache)](https://pypi.org/project/lmcache/)
[![Unit Tests](https://badge.buildkite.com/ce25f1819a274b7966273bfa54f0e02f092c3de0d7563c5c9d.svg)](https://buildkite.com/lmcache/lmcache-unittests)
[![Code Quality](https://github.com/lmcache/LMCache/actions/workflows/code_quality_checks.yml/badge.svg?branch=dev&label=tests)](https://github.com/LMCache/LMCache/actions/workflows/code_quality_checks.yml)
[![Integration Tests](https://badge.buildkite.com/108ddd4ab482a2480999dec8c62a640a3315ed4e6c4e86798e.svg)](https://buildkite.com/lmcache/lmcache-vllm-integration-tests)

<br />

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10841/badge)](https://www.bestpractices.dev/projects/10841)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/LMCache/LMCache/badge)](https://scorecard.dev/viewer/?uri=github.com/LMCache/LMCache)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/LMCache/LMCache/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/w/LMCache/LMCache)](https://github.com/LMCache/LMCache/graphs/commit-activity)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/lmcache)](https://pypi.org/project/lmcache/)
[![YouTube Channel Views](https://img.shields.io/youtube/channel/views/UC58zMz55n70rtf1Ak2PULJA)](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)

---

## LMCache: Accelerate LLM Serving with Efficient KV Cache Reuse

LMCache dramatically reduces latency and boosts throughput for Large Language Model (LLM) serving by intelligently caching and reusing KV caches. [Explore LMCache on GitHub](https://github.com/LMCache/LMCache).

**Key Features:**

*   **Significant Performance Boost:** Reduce Time to First Token (TTFT) and increase throughput, especially in long-context scenarios.
*   **KV Cache Reuse:** Efficiently reuses KV caches of any reused text across instances, saving GPU cycles.
*   **vLLM Integration:** Seamlessly integrates with vLLM, offering features like CPU KVCache offloading and P2P KVCache sharing.
*   **Flexible Storage:** Supports KV cache storage on CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl).
*   **Production Ready:** Supported in the [vLLM production stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve/kserve).
*   **Stable Support for Non-Prefix KV Caches**

## Installation

Install LMCache with a simple pip command:

```bash
pip install lmcache
```

For detailed installation instructions, including troubleshooting and dependencies, consult the [LMCache documentation](https://docs.lmcache.ai/getting_started/installation).

## Getting Started

Start your LMCache journey with our [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the documentation.

## Documentation

Comprehensive documentation is available online: [LMCache Documentation](https://docs.lmcache.ai/).

## Examples

Explore practical use cases with our [examples](https://github.com/LMCache/LMCache/tree/dev/examples) to understand how LMCache can be applied to your projects.

## Community

*   **Blog:** [LMCache Blog](https://blog.lmcache.ai/)
*   **Join the Community:** [Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3bgx768yd-H8WkOTmPtbxVYJ5nuZ4dmA)
*   **Community Meetings:** Bi-weekly on Tuesdays at 9:00 AM PT.
    *   [Meeting Zoom Link](https://uchicago.zoom.us/j/6603596916?pwd=Z1E5MDRWUSt2am5XbEt4dTFkNGx6QT09)
    *   [Add to Google Calendar](https://calendar.google.com/calendar/u/0/r?cid=Y19mNGY2ZmMwZjUxMWYyYTZmZmE1ZTVlMGI2Yzk2NmFmZjNhM2Y4ODZiZmU5OTU5MDJlMmE3ZmUyOGZmZThlOWY5QGdyb3VwLmNhbGVuZGFyLmdvb2dsZS5jb20)
    *   Meeting notes are available [here](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow).
    *   Recordings of meetings can be found on the [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA).
*   **Stay Informed:** [Interest Form](https://forms.gle/MHwLiYDU6kcW3dLj7) | [Newsletter Signup](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter)
*   **Contact:** [Email](mailto:contact@lmcache.ai)

## Contributing

We welcome contributions! Review the [Contributing Guide](CONTRIBUTING.md) for details on how to contribute.  See also [[Onboarding] Welcoming contributors with good first issues!](https://github.com/LMCache/LMCache/issues/627)

## Citation

If you use LMCache in your research, please cite the following papers:

```bibtex
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

LMCache is licensed under the Apache License 2.0.  See the [LICENSE](LICENSE) file for details.