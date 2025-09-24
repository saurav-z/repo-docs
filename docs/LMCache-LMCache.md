<div align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/LMCache/LMCache/dev/asset/logo.png" width="720" alt="LMCache Logo">
  </p>
</div>

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

## LMCache: Accelerate LLM Serving with Intelligent KV Cache Management

LMCache drastically reduces Time-To-First-Token (TTFT) and boosts throughput for Large Language Models (LLMs) by intelligently caching and reusing KV caches.  [**Explore the LMCache Repository**](https://github.com/LMCache/LMCache)

**Key Features:**

*   **Enhanced Performance:** Significantly reduces response times and optimizes GPU utilization.
*   **vLLM Integration:** Seamlessly integrates with vLLM v1 for high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Flexible Caching:** Supports non-prefix KV caches, enabling caching of arbitrary reused text.
*   **Versatile Storage:** Offers storage options including CPU, Disk, and NIXL, ensuring flexibility in deployment.
*   **Production Ready:** Supported in vLLM production stack, llm-d and KServe.
*   **Easy Installation:** Simple pip installation for quick setup.

## What is LMCache?

LMCache is an LLM serving engine extension designed to significantly improve performance by caching the KV caches of reused text across various locations. This approach saves precious GPU resources and reduces user response delay, especially in long-context scenarios.  It combines with vLLM to achieve substantial delay savings and GPU cycle reductions in various LLM applications, including multi-round QA and RAG.

## Getting Started

1.  **Installation:**

    ```bash
    pip install lmcache
    ```
    Ensure you have a Linux NVIDIA GPU platform. Refer to the [detailed installation instructions](https://docs.lmcache.ai/getting_started/installation) for specific setup steps and troubleshooting.

2.  **Quickstart:** Get started with our [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/)

## Resources

*   **Documentation:** [LMCache Documentation](https://docs.lmcache.ai/)
*   **Blog:** [LMCache Blog](https://blog.lmcache.ai/)
*   **Examples:** [LMCache Examples](https://github.com/LMCache/LMCache/tree/dev/examples)
*   **Community:**
    *   [Join Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3bgx768yd-H8WkOTmPtbxVYJ5nuZ4dmA)
    *   [Interest Form](https://forms.gle/MHwLiYDU6kcW3dLj7)
    *   [Roadmap](https://github.com/LMCache/LMCache/issues/1253)
    *   [LMCache website](https://lmcache.ai/)
    *   [Contact us](mailto:contact@lmcache.ai)

## Enterprise Deployment

For enterprise-scale deployment, check out vLLM [Production Stack](https://github.com/vllm-project/production-stack).

## Community

*   **Community Meeting:** Bi-weekly meetings are held on Tuesdays at 9:00 AM PT - [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
*   **Meeting Notes:** Summaries of standups, discussion, and action items are available in this [document](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow)
*   **Meeting Recordings:** Recordings of meetings are available on the [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA).

## Contributing

We welcome contributions! See the [Contributing Guide](CONTRIBUTING.md) for details.

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

## Social Media

*   [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   [Twitter](https://x.com/lmcache)
*   [YouTube](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the [Apache License 2.0](LICENSE).