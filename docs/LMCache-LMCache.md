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

## LMCache: Accelerate LLM Performance with Efficient KV Cache Reuse

LMCache is an LLM serving engine extension designed to significantly boost performance by intelligently caching and reusing KV caches.  Check out the [original repo](https://github.com/LMCache/LMCache) for more details.

**Key Features:**

*   **Enhanced Speed & Throughput:** Reduces Time-To-First-Token (TTFT) and increases throughput, especially in long-context scenarios.
*   **KV Cache Reuse:** Reuses KV caches of any reused text across serving engine instances.
*   **vLLM Integration:** Seamlessly integrates with vLLM v1 for high-performance CPU KV cache offloading, disaggregated prefill, and P2P KV cache sharing.
*   **Broad Support:** Supported in the vLLM production stack, llm-d, and KServe.
*   **Non-Prefix KV Cache Support:**  Offers stable support for non-prefix KV caches, allowing for more flexible caching strategies.
*   **Flexible Storage:** Supports caching on CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl).
*   **Easy Installation:** Simple installation via pip.

## Installation

Install LMCache easily using pip:

```bash
pip install lmcache
```

For detailed installation instructions, especially if you're not using the latest vLLM version or a different serving engine, see the [documentation](https://docs.lmcache.ai/getting_started/installation).  The documentation resolves potential "undefined symbol" or torch mismatch issues.

## Getting Started

Begin your LMCache journey with the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in our documentation.

## Documentation

Explore comprehensive documentation for LMCache at [https://docs.lmcache.ai/](https://docs.lmcache.ai/).

## Examples

Explore practical applications with our [examples](https://github.com/LMCache/LMCache/tree/dev/examples) to address various use cases.

## Community and Support

*   **Blog:** [https://blog.lmcache.ai/](https://blog.lmcache.ai/)
*   **Join our Slack:** [https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q)
*   **Interest Form:** [https://forms.gle/MHwLiYDU6kcW3dLj7](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   **Website:** [https://lmcache.ai/](https://lmcache.ai/)
*   **Email:** [mailto:contact@lmcache.ai](mailto:contact@lmcache.ai)
*   **Community Meetings:** Bi-weekly on Tuesdays at 9:00 AM PT - [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download).  Meeting notes are available [here](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow), and recordings are on the [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA).

## Contributing

We welcome contributions!  See our [Contributing Guide](CONTRIBUTING.md) and the [[Onboarding] Welcoming contributors with good first issues!](https://github.com/LMCache/LMCache/issues/627) for details.

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

*   **LinkedIn:** [https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   **Twitter:** [https://x.com/lmcache](https://x.com/lmcache)
*   **YouTube:** [https://www.youtube.com/@LMCacheTeam](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under Apache License 2.0. See the [LICENSE](LICENSE) file for details.
```
Key improvements and SEO considerations:

*   **Concise Hook:** The one-sentence hook directly highlights LMCache's primary benefit.
*   **Clear Headings:**  Uses semantic headings to organize the content.
*   **Keyword Optimization:** Includes relevant keywords such as "LLM", "KV Cache", "TTFT", "throughput", "vLLM", "serving engine" and "RAG".
*   **Bulleted Lists:** Key features are presented in a clear, easy-to-scan bulleted list, making it easier for users to grasp LMCache's capabilities.
*   **Strong Call to Actions:** Includes clear calls to action to encourage user engagement (e.g., "Explore comprehensive documentation," "Begin your LMCache journey," "We welcome contributions!").
*   **Detailed Installation:** The installation section is kept simple, with a link to the more detailed install guide in the docs.
*   **Contextual Links:**  Hyperlinks are used thoughtfully, linking to relevant resources and documentation.
*   **Community Focus:**  Emphasizes community involvement and provides links to the community meeting, slack, etc.
*   **Citation Section:**  Clearly displays the citation information, essential for researchers.
*   **Social Media Links:** Includes links to social media profiles, increasing visibility.
*   **License Information:** Maintains license visibility.