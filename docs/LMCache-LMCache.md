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

## LMCache: Accelerate LLM Serving by Reusing KV Caches

LMCache dramatically reduces the time-to-first-token (TTFT) and increases throughput for LLMs, especially in long-context scenarios, and you can explore the [LMCache repository on GitHub](https://github.com/LMCache/LMCache) for more details.

**Key Features:**

*   **LLM Serving Acceleration:** Drastically reduces TTFT and boosts throughput for Large Language Models.
*   **KV Cache Reuse:** Stores and reuses KV caches of reusable text across GPU, CPU DRAM, and local disk for significant performance gains.
*   **vLLM Integration:** Seamlessly integrates with vLLM v1, including high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Non-Prefix Cache Support:** Provides stable support for non-prefix KV caches.
*   **Storage Options:** Supports caching on CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl).
*   **Production Ready:** Officially supported in the [vLLM production stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve/kserve).

**Why Use LMCache?**

LMCache empowers developers to achieve substantial delay savings (3-10x) and GPU cycle reduction in various LLM applications, including multi-round QA and RAG.

<img src="https://github.com/user-attachments/assets/86137f17-f216-41a0-96a7-e537764f7a4c" alt="performance">

## Installation

Install LMCache easily using pip:

```bash
pip install lmcache
```

Note: Works on Linux NVIDIA GPU platforms. For detailed instructions, see the [installation documentation](https://docs.lmcache.ai/getting_started/installation).

## Getting Started

Discover how to use LMCache with our [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/).

## Documentation

Comprehensive documentation is available at [docs.lmcache.ai](https://docs.lmcache.ai/). Stay updated with the latest developments via our [blog](https://blog.lmcache.ai/).

## Examples

Explore practical applications and use cases in our [examples](https://github.com/LMCache/LMCache/tree/dev/examples).

## Connect with Us

*   [Interest Form](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   [Newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter)
*   [Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ)
*   [Website](https://lmcache.ai/)
*   [Email](contact@lmcache.ai)

## Community Meetings

Join our weekly community meetings to discuss LMCache development and usage:

*   Meetings **alternate weekly** between these two times:
    *   Tuesdays at 9:00 AM PT – [Add to Calendar](https://drive.google.com/file/d/15Xz8-LtpBQ5QgR7KrorOOyfuohCFQmwn/view?usp=drive_link)
    *   Tuesdays at 6:30 PM PT – [Add to Calendar](https://drive.google.com/file/d/1WMZNFXV24kWzprDjvO-jQ7mOY7whqEdG/view?usp=drive_link)

*   Meeting notes are available [here](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow).
*   Recordings of meetings are on the [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA).

## Contributing

We welcome contributions!  See the [Contributing Guide](CONTRIBUTING.md) for details.

## Citation

If you use LMCache, please cite our research papers:

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

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file.
```
Key improvements and SEO considerations:

*   **Clear and Concise Hook:** The one-sentence introduction clearly states LMCache's value proposition.
*   **Keyword Optimization:** Includes relevant keywords like "LLM," "TTFT," "throughput," "KV cache," and "Large Language Models" throughout the document.
*   **Structured Headings:** Uses clear and descriptive headings for easy readability and SEO ranking.
*   **Bulleted Lists:** Highlights key features, making them easy to scan.
*   **Strong Call to Action (CTA):** Encourages readers to install, explore examples, and connect with the community.
*   **Internal Linking:** Links within the README to other sections, documents and related projects for improved navigation.
*   **External Links:** Direct links to the documentation, blog, and relevant project resources.
*   **SEO-Friendly Structure:**  Uses proper HTML formatting (although within Markdown) for headings and lists, improving readability and search engine indexing.
*   **Mobile Responsiveness:** The use of relative sizing and clean formatting makes it easy to read on any device.