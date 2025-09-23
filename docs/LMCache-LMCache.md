# LMCache: Accelerate LLM Serving with Intelligent KV Cache Management

**LMCache dramatically reduces latency and improves throughput for Large Language Models by intelligently caching and reusing key-value (KV) caches.** ([Original Repository](https://github.com/LMCache/LMCache))

---

## Key Features

*   **Reduced TTFT & Increased Throughput:** Significantly lowers Time-To-First-Token (TTFT) and boosts throughput, especially in long-context scenarios.
*   **KV Cache Reusability:**  Reuses KV caches for any reused text, not just prefixes, across different serving engine instances.
*   **vLLM Integration:** Seamlessly integrates with vLLM for enhanced performance and features such as high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Production Ready:** Supports vLLM production stack, llm-d and KServe.
*   **Flexible Storage Options:** Offers diverse storage options including CPU, Disk and NiXL.
*   **Non-Prefix Cache Support:** Offers stable support for non-prefix KV caches.
*   **Easy Installation:** Simple pip installation.

## Installation

Install LMCache using pip:

```bash
pip install lmcache
```

For detailed installation instructions, especially if you are not using the latest stable version of vLLM, refer to the [documentation](https://docs.lmcache.ai/getting_started/installation)

## Getting Started

Explore our [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) to quickly get started with LMCache.

## Documentation

Comprehensive documentation is available at [https://docs.lmcache.ai/](https://docs.lmcache.ai/), including a [blog](https://blog.lmcache.ai/).

## Examples

Find practical demonstrations of LMCache in action within our [examples](https://github.com/LMCache/LMCache/tree/dev/examples) directory.

## Connect with Us

*   **Interest Form:** [https://forms.gle/MHwLiYDU6kcW3dLj7](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   **Newsletter:** [https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter)
*   **Slack:** [https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3bgx768yd-H8WkOTmPtbxVYJ5nuZ4dmA](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3bgx768yd-H8WkOTmPtbxVYJ5nuZ4dmA)
*   **Website:** [https://lmcache.ai/](https://lmcache.ai/)
*   **Email:** [contact@lmcache.ai](mailto:contact@lmcache.ai)

## Community Meetings

Join our bi-weekly community meetings to discuss LMCache development and usage.

*   **Schedule:** Tuesdays at 9:00 AM PT - [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
*   **Meeting Notes:** [https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow)
*   **Recordings:** [YouTube LMCache Channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)

## Contributing

We welcome contributions!  Please review our [Contributing Guide](CONTRIBUTING.md) and [good first issues](https://github.com/LMCache/LMCache/issues/627).

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

*   **LinkedIn:** [https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   **Twitter:** [https://x.com/lmcache](https://x.com/lmcache)
*   **YouTube:** [https://www.youtube.com/@LMCacheTeam](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the Apache License 2.0.  See the [LICENSE](LICENSE) file for details.