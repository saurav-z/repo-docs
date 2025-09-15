# LMCache: Accelerate LLM Serving with Intelligent KV Cache Management

**LMCache dramatically reduces the time-to-first-token (TTFT) and boosts throughput for your large language models by intelligently managing and reusing KV caches.** ([View on GitHub](https://github.com/LMCache/LMCache))

---

## Key Features

*   🚀 **Enhanced Performance:** Drastically reduces TTFT and increases throughput, especially for long-context and RAG applications.
*   💾 **KV Cache Reusability:** Efficiently reuses KV caches of any reused text across instances, saving precious GPU resources.
*   🤝 **Seamless Integration:**  Works with vLLM and is supported in the vLLM production stack, llm-d, and KServe.
*   💾 **Flexible Storage:** Supports KV cache storage on CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl).
*   ✅ **Stable Non-Prefix KV Cache Support:**  Offers stable support for non-prefix KV caches for diverse use cases.
*   📦 **Easy Installation:** Simple pip installation for quick setup.

---

## Installation

```bash
pip install lmcache
```

For detailed instructions and troubleshooting, consult the [Installation Guide](https://docs.lmcache.ai/getting_started/installation) in the documentation.

---

## Getting Started

Explore the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) to rapidly integrate LMCache into your LLM serving pipeline.

---

## Documentation

Comprehensive documentation is available to help you understand and leverage LMCache's full capabilities:

*   [Documentation](https://docs.lmcache.ai/)
*   [Blog](https://blog.lmcache.ai/)

---

## Examples

Learn how to use LMCache with hands-on examples: [Examples](https://github.com/LMCache/LMCache/tree/dev/examples).

---

## Community

*   **Join our Community Meeting:** Tuesdays at 9:00 AM PT - [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
*   **Meeting Notes:** [Meeting Notes](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow)
*   **YouTube Channel:** [LMCache YouTube Channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)
*   **Join Slack:** [Join Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3bgx768yd-H8WkOTmPtbxVYJ5nuZ4dmA)
*   **Interest Form:** [Interest Form](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   **Check out LMCache Website:** [LMCache Website](https://lmcache.ai/)
*   **Drop an Email:** [contact@lmcache.ai](mailto:contact@lmcache.ai)

---

## Contributing

We welcome contributions! Please review the [Contributing Guide](CONTRIBUTING.md) for details.

---

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

---

## Socials

*   [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   [Twitter](https://x.com/lmcache)
*   [YouTube](https://www.youtube.com/@LMCacheTeam)

---

## License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.