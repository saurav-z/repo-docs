# verl: Train LLMs Efficiently with Reinforcement Learning

verl is an open-source RL training library for large language models (LLMs), offering flexible, efficient, and production-ready solutions for LLM training and alignment. [Explore the verl repository](https://github.com/volcengine/verl).

[![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl)](https://github.com/volcengine/verl/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/verl_project)](https://twitter.com/verl_project)
[![Slack](https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp)](https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA)
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)
<a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red"></a>
<a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>

## Key Features

*   **Flexible RL Algorithms:** Easily extend and implement various RL algorithms like PPO, GRPO, and DAPO with a hybrid-controller programming model.
*   **Seamless LLM Integration:** Integrate with existing LLM infrastructures like FSDP, Megatron-LM, vLLM, and SGLang using modular APIs.
*   **Efficient Resource Utilization:** Supports flexible device mapping for optimal resource utilization and scalability across different cluster sizes.
*   **Hugging Face Compatibility:** Ready integration with popular Hugging Face models.
*   **State-of-the-art Throughput:** Integrates SOTA LLM training and inference engines for efficient training and inference.
*   **Accelerated Training:** Optimized actor model resharding with 3D-HybridEngine to reduce communication overhead.
*   **Model & Dataset Support:** Compatible with a variety of models, including Qwen-3, Llama3.1, Gemma2, and support for Supervised Fine-tuning.
*   **Multi-Modality Support:** Supports Vision-Language Models (VLMs) and Multi-Modal RL.

## Key Updates

*   **[July 2025]** ReTool recipe is fully open-sourced: [Blog](https://www.notion.so/verl-reTool-recipe-Using-multi-round-conversations-and-code-sandboxing-to-improve-the-math-of-large-23a8b5b7feba80b386b2e5b5e3c1cde0)
*   **[July 2025]** First verl Meetup at ICML Vancouver on July 16th! [Join us](https://lu.ma/0ek2nyao) if you are at ICML!
*   **[June 2025]** verl with Megatron backend enables large MoE models such as [DeepSeek-671b and Qwen3-236b](https://verl.readthedocs.io/en/latest/perf/dpsk.html).
*   **[April 2025]** VAPO paper covers our latest RL method for reasoning models. [Paper](https://arxiv.org/pdf/2504.05118)
*   **[March 2025]** verl v0.3.0.post1 is released with ~1.4x speedup. See [release note](https://github.com/volcengine/verl/releases/)

For more news, see the full README.

## Getting Started

*   [Documentation](https://verl.readthedocs.io/en/latest/index.html)
*   [Installation](https://verl.readthedocs.io/en/latest/start/install.html)
*   [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
*   [PPO Example](https://verl.readthedocs.io/en/latest/algo/ppo.html)
*   [GRPO Example](https://verl.readthedocs.io/en/latest/algo/grpo.html)

## Community Resources

*   **Blogs and Tutorials:** [Explore various blogs and tutorials](https://github.com/volcengine/verl#blogs-from-the-community) from the community.
*   **Performance Tuning:** [Optimize performance](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) with our performance tuning guide.
*   **AMD Support:** Access guides for [AMD support](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_build_dockerfile_page.rst)

## Citation

If you use verl in your research, please cite the HybridFlow paper:
```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

## Acknowledgements

verl is inspired by several existing frameworks and has been adopted and contributed by a wide range of organizations and individuals. See the original README for a complete list.

## Contribution

Contributions are welcome! See the [contribution guide](CONTRIBUTING.md) for more information.

## About ByteDance Seed Team

Learn more about the ByteDance Seed Team and their work:
*   [Website](https://team.doubao.com/)
*   [WeChat](https://github.com/user-attachments/assets/469535a8-42f2-4797-acdf-4f7a1d4a0c3e)
*   [Xiaohongshu](https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search)
*   [Zhihu](https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/)

Interested in RL for agents? Contact us: [haibin.lin@bytedance.com](mailto:haibin.lin@bytedance.com)
```
Key improvements and explanations:

*   **SEO Optimization:** Title and headings use keywords like "Reinforcement Learning," "LLMs," "RL," and "LLM training."  The summary is concise and emphasizes key benefits.
*   **One-Sentence Hook:** "verl is an open-source RL training library for large language models (LLMs), offering flexible, efficient, and production-ready solutions for LLM training and alignment." Grabs attention immediately.
*   **Clear Structure:**  Uses headings for readability and easy navigation: Key Features, Getting Started, Community Resources, etc.
*   **Bulleted Key Features:** Makes the benefits easy to scan. The features are also worded to be more appealing.
*   **Concise Descriptions:** Each feature has a brief and compelling explanation.
*   **Actionable Links:**  Provides direct links to documentation and quickstart guides.
*   **Clear "About" Sections:** Includes specific information about the ByteDance Seed Team.
*   **Removed Redundancy:** Removed unnecessary introductory phrases ("Hi everyone!")
*   **Consolidated News:** Presented news items in a more readable list.
*   **Updated News:** Maintained the latest news from the provided README.
*   **Contact Information:** Adds a call to action for potential contributors.
*   **Concise:** Trimmed extraneous information for improved readability.
*   **Emphasis on Benefits:** Highlights what users *get* from verl, not just what it *is*.