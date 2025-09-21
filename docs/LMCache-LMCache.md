# LMCache: Accelerate LLM Serving with Intelligent Caching

**LMCache significantly reduces the time-to-first-token (TTFT) and boosts throughput for LLMs, especially in long-context scenarios.**  Learn more at the [LMCache GitHub repository](https://github.com/LMCache/LMCache).

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

**Key Features:**

*   **Reduced Latency:** Significantly lowers TTFT for faster LLM response times.
*   **Increased Throughput:** Improves the number of requests that can be handled concurrently.
*   **KV Cache Reuse:**  Efficiently reuses KV caches across instances and for non-prefix matches.
*   **vLLM Integration:** Optimized integration with vLLM for enhanced performance, including CPU KVCache offloading, disaggregated prefill and P2P KVCache sharing.
*   **Flexible Storage:** Supports KV cache storage on GPU, CPU (DRAM), and disk.
*   **Production-Ready:** Compatible with the vLLM production stack, llm-d and KServe for enterprise deployments.
*   **Stable Non-Prefix Support:**  Reliably caches and reuses KV caches regardless of prefix.

🔥 **NEW: For enterprise-scale deployment of LMCache and vLLM, please check out vLLM [Production Stack](https://github.com/vllm-project/production-stack). LMCache is also officially supported in [llm-d](https://github.com/llm-d/llm-d/) and [KServe](https://github.com/kserve/kserve)!**

## Benefits

*   **Accelerated LLM Applications:** Achieve 3-10x delay savings and GPU cycle reduction.
*   **Optimized Resource Utilization:**  Reduce GPU usage and improve overall efficiency.
*   **Enhanced User Experience:** Deliver faster and more responsive LLM-powered applications.

![performance](https://github.com/user-attachments/assets/86137f17-f216-41a0-96a7-e537764f7a4c)

## Installation

Install LMCache via pip:

```bash
pip install lmcache
```

For detailed instructions and troubleshooting, consult the [Installation Guide](https://docs.lmcache.ai/getting_started/installation) in the documentation.

## Getting Started

Explore the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) to quickly get up and running with LMCache.

## Documentation

Comprehensive documentation is available at [docs.lmcache.ai](https://docs.lmcache.ai/), including guides, tutorials, and API references.

## Examples

Explore practical examples in the [examples](https://github.com/LMCache/LMCache/tree/dev/examples) directory to see how LMCache can be applied to various use cases.

## Connect with Us

*   [Join Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3bgx768yd-H8WkOTmPtbxVYJ5nuZ4dmA)
*   [Interest Form](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   [LMCache website](https://lmcache.ai/)
*   [Newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter)
*   [Email](mailto:contact@lmcache.ai)
*   [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   [Twitter](https://x.com/lmcache)
*   [YouTube](https://www.youtube.com/@LMCacheTeam)

## Community Meetings

Join the bi-weekly community meetings to discuss LMCache:

*   **Schedule:** Tuesdays at 9:00 AM PT - [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
*   **Meeting Notes:** [Google Docs](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow)
*   **Recordings:** [YouTube Channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)

## Contributing

We welcome contributions!  See the [Contributing Guide](CONTRIBUTING.md) and the [Onboarding](https://github.com/LMCache/LMCache/issues/627) issue for details.

## Citation

If you use LMCache, please cite our research:

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

## License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.