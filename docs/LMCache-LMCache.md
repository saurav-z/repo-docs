<div align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/LMCache/LMCache/dev/asset/logo.png" width="720" alt="lmcache logo">
  </p>
  
  <!-- Badges - keep these, they are useful -->
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
</div>

# LMCache: Accelerate LLM Serving with Intelligent KV Cache Management

LMCache significantly boosts LLM performance by efficiently caching and reusing key-value (KV) caches, reducing latency and saving valuable GPU resources.

## Key Features

*   **Enhanced Performance:** Dramatically reduces Time-to-First-Token (TTFT) and increases throughput, especially beneficial for long-context LLM applications.
*   **Intelligent KV Cache Reuse:** Caches KV data across various storage locations (GPU, CPU, Disk) to reuse any repeated text, not just prefixes, across any serving engine instance.
*   **vLLM Integration:** Seamlessly integrates with vLLM v1, offering features like high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Broad Compatibility:** Supported by the vLLM production stack, llm-d, and KServe.
*   **Flexible Storage Options:** Supports caching on CPU, Disk, and NIXL for optimal resource utilization.
*   **Non-Prefix Cache Support:** Offers stable support for non-prefix KV caches, enhancing caching flexibility.

## Installation

Install LMCache easily using pip:

```bash
pip install lmcache
```

For detailed instructions, see the [LMCache Documentation](https://docs.lmcache.ai/getting_started/installation).  Requires a Linux NVIDIA GPU platform.

## Getting Started

Explore the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the documentation to begin using LMCache.

## Documentation

Comprehensive documentation is available at [docs.lmcache.ai](https://docs.lmcache.ai/) providing in-depth information and usage guides.  Stay up-to-date with the [LMCache blog](https://blog.lmcache.ai/) for the latest updates and insights.

## Examples

Explore practical use cases with our [examples](https://github.com/LMCache/LMCache/tree/dev/examples) demonstrating how to integrate LMCache into different LLM serving scenarios.

## Community and Resources

*   **Join the Community:** Connect with the LMCache team and other users via the [LMCache Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q).
*   **Stay Informed:** Sign up for the [LMCache Newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter).
*   **Explore the Website:** Visit the [LMCache website](https://lmcache.ai/) for additional information.
*   **Reach Out:** Contact the team directly at [contact@lmcache.ai](mailto:contact@lmcache.ai).
*   **Community Meetings:** Join the bi-weekly [community meetings]( https://uchicago.zoom.us/j/6603596916?pwd=Z1E5MDRWUSt2am5XbEt4dTFkNGx6QT09) (Tuesdays at 9:00 AM PT – [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)) and access meeting notes and recordings.
*   **Roadmap:** Check out the [Roadmap](https://github.com/LMCache/LMCache/issues/1253) to see what's next.

## Contributing

We welcome contributions! Review the [Contributing Guide](CONTRIBUTING.md) to learn how to contribute to LMCache. Check out [[Onboarding] Welcoming contributors with good first issues!](https://github.com/LMCache/LMCache/issues/627) to get started.

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

## Social Media

*   [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   [Twitter](https://x.com/lmcache)
*   [YouTube](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

[**View the original repository on GitHub**](https://github.com/LMCache/LMCache)
```
Key improvements and explanations:

*   **SEO-Optimized Title:**  The title now explicitly states "Accelerate LLM Serving" which is a key search term, making it easier for users to find the project.  The phrase "Intelligent KV Cache Management" is also included for its relevance.
*   **Concise Hook:** The one-sentence hook provides a clear value proposition and immediately grabs the reader's attention, summarizing what LMCache does.
*   **Clear Headings:** Uses headings like "Key Features," "Installation," etc. to structure the content logically and make it easy to scan.  This is essential for readability and SEO.
*   **Bulleted Key Features:** Uses bullet points to highlight key features, making them easy to understand and improving scannability.
*   **Keyword Optimization:** The text uses relevant keywords naturally throughout the description, improving search engine visibility (e.g., "LLM," "KV Cache," "vLLM").
*   **Stronger Summary:** The summary is more concise and directly states the problem LMCache solves (TTFT, throughput) and its benefits.
*   **Call to Action:** The "Community and Resources" section offers clear calls to action, encouraging users to engage.
*   **Community & Social Links:**  The links to social media, documentation, and community resources are retained and easily accessible.
*   **Clean Formatting:** The use of bolding and consistent formatting improves readability.
*   **Links to key resources:** The links to the documentation, examples, and other project resources are clear and easy to find.
*   **Back to original repo link:** Included a clear link at the end for the user to get back to the project's main page.
*   **Removed redundancy:** Cleaned up the language and removed redundant information.
*   **Improved flow and readability:**  The text flows better and is easier to understand.
*   **Kept Useful Badges:** Maintained the useful badges to show project health and status.

This improved README is more informative, user-friendly, and optimized for search engines, helping to attract more users to the LMCache project.