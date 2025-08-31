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

## LMCache: Accelerate LLM Serving with Efficient KV Cache Management

LMCache is a cutting-edge LLM serving engine extension designed to dramatically reduce Time-to-First-Token (TTFT) and boost throughput by intelligently caching KV caches.  Explore the [LMCache repository on GitHub](https://github.com/LMCache/LMCache).

**Key Features:**

*   **Enhanced Performance:** Significantly reduces TTFT and increases throughput, especially in long-context scenarios, improving LLM response times.
*   **KV Cache Reuse:** Stores and reuses KV caches across GPU, CPU DRAM, and local disk, for any reused text within serving engine instances, regardless of the prefix.
*   **vLLM Integration:** Seamlessly integrates with vLLM v1, offering high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Broad Compatibility:** Supports KV cache storage on CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl).
*   **Production Ready:** Officially supported by the [vLLM production stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve/kserve).
*   **Non-Prefix Cache Support:** Provides stable support for non-prefix KV caches.
*   **Easy Installation:** Simple pip installation for quick setup.

## Installation

Install LMCache easily using pip:

```bash
pip install lmcache
```

For detailed installation instructions, especially if you are not using the latest stable vLLM version, refer to the [documentation](https://docs.lmcache.ai/getting_started/installation) to resolve any "undefined symbol" or torch mismatch issues.

## Getting Started

Begin your LMCache journey with our easy-to-follow [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the documentation.

## Documentation

Comprehensive documentation is available online to guide you: [Documentation](https://docs.lmcache.ai/).  Stay updated through our [blog](https://blog.lmcache.ai/).

## Examples

Dive into practical application with our [examples](https://github.com/LMCache/LMCache/tree/dev/examples), demonstrating LMCache in various use cases.

## Community

*   **Join us on Slack:** [Slack Invite](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q)
*   **Community Meetings:**  Bi-weekly meetings are held on Tuesdays at 9:00 AM PT. [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
*   **Meeting Notes:** Summaries of meetings are kept here:  [Meeting Notes](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow)
*   **YouTube Channel:** Watch meeting recordings and more on the [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA).
*   **Interested in Connecting?** Fill out the [interest form](https://forms.gle/mQfQDUXbKfp2St1z7), [sign up for our newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter), or [drop an email](mailto:contact@lmcache.ai)!

## Contributing

We welcome all contributions! Check out the [Contributing Guide](CONTRIBUTING.md) and our [Onboarding issues](https://github.com/LMCache/LMCache/issues/627) for getting started.

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

LMCache is licensed under the Apache License 2.0.  See the [LICENSE](LICENSE) file for details.
```
Key improvements and SEO considerations:

*   **Clear Headline:**  The main headline is now more SEO-friendly, including the keywords "LMCache" and "LLM" and "KV Cache".
*   **Concise Hook:**  A compelling one-sentence hook highlights the core benefit.
*   **Keyword Optimization:**  Keywords like "LLM", "KV Cache", "TTFT", and "throughput" are strategically placed throughout the text.
*   **Bulleted Features:**  Uses bullet points for readability and emphasis on key features.
*   **Structured Headings:** Uses headings to improve readability and SEO.
*   **Call to Action:** Encourages the user to explore the repository.
*   **Links:** Links are included.
*   **Community Engagement:**  Highlights community aspects, which is good for engagement and SEO.