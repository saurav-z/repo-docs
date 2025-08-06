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

---

## LMCache: Accelerate LLM Serving with Efficient KV Cache Management

LMCache is a cutting-edge extension designed to significantly reduce Time-to-First-Token (TTFT) and boost throughput for Large Language Models (LLMs).  Find out more on the [original repository](https://github.com/LMCache/LMCache).

### Key Features

*   **Enhanced Performance:** Drastically reduces response times and optimizes GPU resource utilization by reusing KV caches.
*   **vLLM Integration:** Seamlessly integrates with vLLM v1, offering high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Broad Support:**  Compatible with the vLLM production stack, llm-d, and KServe.
*   **Flexible Cache Management:**  Supports non-prefix KV caches for wider applicability and storage options including CPU, Disk, and NIXL.
*   **Easy Installation:**  Simple pip installation for quick setup.

### Key Benefits

*   **Faster LLM Serving:** Achieve up to 3-10x delay savings and GPU cycle reduction in use cases like multi-round QA and RAG.
*   **Optimized Resource Utilization:**  Efficiently manages KV caches across various locations (GPU, CPU DRAM, Local Disk) to save precious GPU cycles.
*   **Improved User Experience:**  Reduce latency and improve responsiveness for a better user experience.

### Installation

To install LMCache, use pip:

```bash
pip install lmcache
```

Detailed installation instructions are available in the [documentation](https://docs.lmcache.ai/getting_started/installation).

### Getting Started

Explore our [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the documentation to get hands-on with LMCache.

### Documentation

Comprehensive documentation is available at [https://docs.lmcache.ai/](https://docs.lmcache.ai/).

### Examples

Discover practical use cases in our [examples](https://github.com/LMCache/LMCache/tree/dev/examples) to learn how to implement LMCache in various scenarios.

### Stay Connected

*   **Blog:** [https://blog.lmcache.ai/](https://blog.lmcache.ai/)
*   **Documentation:** [https://docs.lmcache.ai/](https://docs.lmcache.ai/)
*   **Join Slack:** [https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q)
*   **Interest Form:** [https://forms.gle/MHwLiYDU6kcW3dLj7](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   **Roadmap:** [https://github.com/LMCache/LMCache/issues/574](https://github.com/LMCache/LMCache/issues/574)
*   **Website:** [https://lmcache.ai/](https://lmcache.ai/)
*   **Email:** contact@lmcache.ai
*   **LinkedIn:** [https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   **Twitter:** [https://x.com/lmcache](https://x.com/lmcache)
*   **YouTube:** [https://www.youtube.com/@LMCacheTeam](https://www.youtube.com/@LMCacheTeam)

### Community Meeting

Join our bi-weekly community meeting for updates and discussions.

*   **Schedule:** Tuesdays at 9:00 AM PT
*   **Meeting Link:** [https://uchicago.zoom.us/j/6603596916?pwd=Z1E5MDRWUSt2am5XbEt4dTFkNGx6QT09](https://uchicago.zoom.us/j/6603596916?pwd=Z1E5MDRWUSt2am5XbEt4dTFkNGx6QT09)
*   **Add to Calendar:** [https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
*   **Meeting Notes:** [https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow)
*   **Recordings:** [https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)

### Contributing

We welcome your contributions!  Please refer to our [Contributing Guide](CONTRIBUTING.md).

### Citations

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

LMCache is licensed under the Apache License 2.0.  See the [LICENSE](LICENSE) file.