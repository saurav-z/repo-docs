<div align="center">
    <h1>verl: Train LLMs Faster with Reinforcement Learning</h1>
    <p>verl is an open-source RL training library by the ByteDance Seed team for efficiently aligning large language models, offering state-of-the-art throughput and flexibility for various RL algorithms.</p>
</div>

[![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl?style=social)](https://github.com/volcengine/verl/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/verl_project?style=social)](https://twitter.com/verl_project)
<a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp"></a>
<a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red"></a>
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)
<a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

## Key Features

*   **Flexible RL Algorithms:** Easily extend and implement various RL algorithms like PPO, GRPO, and more with the hybrid-controller programming model.
*   **Seamless Integration:** Integrate with existing LLM frameworks such as FSDP, Megatron-LM, vLLM, and SGLang with modular APIs.
*   **Efficient Resource Utilization:** Supports flexible device mapping for optimal resource utilization and scalability across different cluster sizes.
*   **Hugging Face Compatibility:** Ready-to-use with popular Hugging Face models.
*   **High Throughput:** Benefit from state-of-the-art LLM training and inference engine integrations.
*   **3D-HybridEngine for Resharding:** Eliminates memory redundancy and reduces communication overhead.

## News

*   **[July 2025]** - verl will be at ICML, AWS AI Hours, and Agent for SWE meetup.
*   **[June 2025]** - verl supports large MoE models and will be at PyTorch Day China.
*   **[April 2025]** - Seed-Thinking-v1.5 tech report released.
*   **[March 2025]** - DAPO achieves SOTA on AIME 2024, trained with verl.
*   [More News...](https://github.com/volcengine/verl#news)

## Getting Started

*   **Documentation:** [https://verl.readthedocs.io/en/latest/index.html](https://verl.readthedocs.io/en/latest/index.html)
*   **Installation:** [https://verl.readthedocs.io/en/latest/start/install.html](https://verl.readthedocs.io/en/latest/start/install.html)
*   **Quickstart:** [https://verl.readthedocs.io/en/latest/start/quickstart.html](https://verl.readthedocs.io/en/latest/start/quickstart.html)
*   **Programming Guide:** [https://verl.readthedocs.io/en/latest/hybrid_flow.html](https://verl.readthedocs.io/en/latest/hybrid_flow.html)
*   **PPO in verl:** [https://verl.readthedocs.io/en/latest/algo/ppo.html](https://verl.readthedocs.io/en/latest/algo/ppo.html)
*   **GRPO in verl:** [https://verl.readthedocs.io/en/latest/algo/grpo.html](https://verl.readthedocs.io/en/latest/algo/grpo.html)
*   [And more...](https://github.com/volcengine/verl#getting-started)

## Performance Tuning & Advanced Features

*   **Performance Tuning:** Comprehensive [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) to optimize performance.
*   **vLLM & SGLang Integration:** Support for vLLM >= 0.8.2 and latest SGLang.
*   **FSDP2 & ROCm Kernel:** Full support for FSDP2 and AMD ROCm.
*   [And more...](https://github.com/volcengine/verl#getting-started)

## Citation

If you find this project useful, please cite the HybridFlow paper:

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

## Contribution

See [contributions guide](CONTRIBUTING.md)

## About ByteDance Seed Team

[Website](https://team.doubao.com/)
<a href="https://team.doubao.com/"><img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
<a href="https://github.com/user-attachments/assets/469535a8-42f2-4797-acdf-4f7a1d4a0c3e">
<img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
<a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
<img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
<a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
<img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
---

We are HIRING! Send us an [email](mailto:haibin.lin@bytedance.com) if you are interested in internship/FTE opportunities in RL for agents.

---
**[Back to Top](https://github.com/volcengine/verl)**
```
Key improvements and explanations:

*   **SEO Optimization:** The title includes relevant keywords like "Train LLMs," "Reinforcement Learning," and "Large Language Models," and the description is crafted to be SEO-friendly.
*   **Concise Hook:** The one-sentence hook immediately grabs the reader's attention and clearly explains verl's primary function.
*   **Clear Headings:** The content is organized with clear, descriptive headings for better readability and SEO.
*   **Bulleted Key Features:** Highlights the key features of verl with bullet points, improving scannability.
*   **Condensed and Rephrased Content:** The text is more concise and the language is improved.
*   **Call to Action:** Clear calls to action, directing users to relevant resources.
*   **Links:**  Maintains and updates links to documentation, relevant papers and community resources.
*   **Removed Redundancy:** Removed repeated information.
*   **Updated News:** News section is shorter and more to-the-point.
*   **Contact Info:** Keeps the hiring contact email at the bottom
*   **"Back to Top" Link:** This improves navigation within the README.
*   **Social Media Badges:** Kept social badges and updated the format for a modern look.
*   **Reformatted Bibtex** Put it into a code block for readability and consistency.