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

## LMCache: Accelerate LLM Inference with Intelligent Caching

LMCache dramatically reduces latency and increases throughput for Large Language Models by intelligently caching and reusing KV caches.  Learn more on the [LMCache GitHub repository](https://github.com/LMCache/LMCache).

### Key Features

*   **Improved Performance:** Reduces Time-to-First-Token (TTFT) and boosts throughput, especially for long-context LLMs.
*   **Efficient Caching:** Caches KV caches across GPU, CPU DRAM, and local disk for reuse.
*   **Flexible Cache Reuse:** Reuses KV caches for any reused text, not just prefixes, across different instances.
*   **vLLM Integration:** Seamlessly integrates with vLLM for enhanced performance (3-10x delay savings).
*   **Storage Options:** Supports caching on CPU, Disk, and NIXL.
*   **Production Ready:** Supported in the vLLM production stack, llm-d, and KServe.

### Summary

LMCache is an LLM serving engine extension that dramatically reduces TTFT and increases throughput, especially in long-context scenarios. By storing KV caches across various locations (GPU, CPU DRAM, Local Disk), LMCache reuses the KV caches of any reused text, saving GPU cycles and user response time.  Combine LMCache with vLLM for significant delay savings and GPU cycle reduction in various LLM use cases, including multi-round QA and RAG.

### Installation

Install LMCache using pip:

```bash
pip install lmcache
```

For detailed installation instructions, refer to the [documentation](https://docs.lmcache.ai/getting_started/installation).  Works on Linux NVIDIA GPU platforms.

### Getting Started

Explore the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) to begin using LMCache.

### Documentation

Comprehensive documentation is available at [https://docs.lmcache.ai/](https://docs.lmcache.ai/).

### Examples

Find practical examples demonstrating various use cases in the [examples](https://github.com/LMCache/LMCache/tree/dev/examples) directory.

### Community

*   **Blog:** [https://blog.lmcache.ai/](https://blog.lmcache.ai/)
*   **Slack:** [Join Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q)
*   **Interest Form:** [Interest Form](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   **Roadmap:** [Roadmap](https://github.com/LMCache/LMCache/issues/1253)

### Community Meeting

Join our bi-weekly community meetings:

*   **Schedule:** Tuesdays at 9:00 AM PT
*   **Zoom Link:** [Community meeting]( https://uchicago.zoom.us/j/6603596916?pwd=Z1E5MDRWUSt2am5XbEt4dTFkNGx6QT09)
*   **Add to Calendar:**  [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
*   **Meeting Notes:** [Meeting Notes](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow)
*   **Recordings:** [YouTube Channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)

### Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) and the [[Onboarding] Welcoming contributors with good first issues!](https://github.com/LMCache/LMCache/issues/627) for details.

### Citation

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

### Social Media

*   **LinkedIn:** [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   **Twitter:** [Twitter](https://x.com/lmcache)
*   **YouTube:** [Youtube](https://www.youtube.com/@LMCacheTeam)

### License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.