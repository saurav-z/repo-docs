<div align="center">
  <img src="docs/source-en/_static/svg/logo_white.svg" alt="RLinf-logo" width="600"/>
</div>

<div align="center">
<a href="https://arxiv.org/abs/2509.15965"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv"></a>
<a href="https://huggingface.co/RLinf"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
<a href="https://rlinf.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
<a href="https://rlinf.readthedocs.io/zh-cn/latest/"><img src="https://img.shields.io/badge/中文文档-red?logo=readthedocs"></a>
<a href="https://deepwiki.com/RLinf/RLinf"><img src="https://img.shields.io/badge/Ask%20DeepWiki-1DA1F2?logo=databricks&logoColor=white&color=00ADEF" alt="Ask DeepWiki"></a>
<a href="https://github.com/RLinf/misc/blob/main/pic/wechat.jpg?raw=true"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

<h1 align="center">
  RLinf: Revolutionizing Agentic AI with Flexible and Efficient Reinforcement Learning
</h1>

RLinf is an open-source infrastructure for post-training foundation models using reinforcement learning, designed to be a robust backbone for next-generation training and agentic AI. [Explore RLinf on GitHub](https://github.com/RLinf/RLinf).

## Key Features

*   **Macro-to-Micro Flow (M2Flow):** Decouples logical workflow construction from physical execution for improved efficiency and programmability.
*   **Flexible Execution Modes:** Supports collocated, disaggregated, and hybrid modes for optimal resource utilization.
*   **Auto-Scheduling Strategy:** Dynamically selects the most suitable execution mode, eliminating manual resource allocation.
*   **Embodied Agent Support:**
    *   Fast adaptation support for mainstream VLA models like OpenVLA and π₀.
    *   Integration with CPU & GPU-based simulators via standardized RL interfaces (ManiSkill3, LIBERO).
    *   Enables RL fine-tuning of π₀ and π₀.₅ models.
*   **Performance Advantages:**
    *   **120%+** throughput improvement in hybrid mode with fine-grained pipelining compared to other frameworks.
    *   Automatic online scaling strategy for faster training and on-policy algorithm preservation.
*   **Versatile Integrations:**
    *   Seamless integration with FSDP + Hugging Face for rapid prototyping and new models.
    *   Optimized for large-scale training via Megatron + SGLang for maximum efficiency.
    *   Adaptive communication via asynchronous communication channels.
*   **Built-in RL Algorithms:** Includes support for popular methods like PPO, GRPO, DAPO, and Reinforce++.

## Main Results

### Embodied Intelligence

<div align="center">
<table>
  <tr>
    <th colspan="5" style="text-align:center;"><strong>OpenVLA-OFT model results on ManiSkill3</strong></th>
  </tr>
  <tr>
    <th>Model</th>
    <th>Vision</th>
    <th>Semantic</th>
    <th>Position</th>
    <th>Average</th>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">rl4vla</a></td>
    <td>76.6%</td>
    <td>75.4%</td>
    <td>77.6%</td>
    <td>76.1%</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-ManiSkill3-25ood"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">GRPO-OpenVLA-OFT</td>
    <td><strong>84.6%</strong></td>
    <td>51.6%</td>
    <td>42.9%</td>
    <td>61.5%</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">PPO-OpenVLA-OFT</td>
    <td>80.5%</td>
    <td>56.6%</td>
    <td>56.1%</td>
    <td>64.5%</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/RLinf/RLinf-OpenVLA-PPO-ManiSkill3-25ood"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">PPO-OpenVLA</td>
    <td>82.0%</td>
    <td><strong>80.6%</strong></td>
    <td><strong>89.3%</strong></td>
    <td><strong>82.2%</strong></td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/RLinf/RLinf-OpenVLA-GRPO-ManiSkill3-25ood"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">GRPO-OpenVLA</td>
    <td>74.7%</td>
    <td>74.4%</td>
    <td>81.6%</td>
    <td>75.5%</td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="6" style="text-align:center;"><strong>OpenVLA-OFT model results on LIBERO</strong></th>
  </tr>
  <tr>
    <th>Model</th>
    <th><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-spatial">Spatial</a></th>
    <th><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-goal">Goal</a></th>
    <th><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-object">Object</a></th>
    <th><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-long">Long</a></th>
    <th>Average</th>
  </tr>
  <tr>
    <td>OpenVLA-OFT-SFT (one-shot)</td>
    <td>56.5%</td>
    <td>45.6%</td>
    <td>25.6%</td>
    <td>9.7%</td>
    <td>34.4%</td>
  </tr>
  <tr>
    <td>OpenVLA-OFT-RLinf</td>
    <td><strong>99.0%</strong></td>
    <td><strong>99.0%</strong></td>
    <td><strong>99.0%</strong></td>
    <td><strong>94.4%</strong></td>
    <td><strong>97.9%</strong></td>
  </tr>
  <tr>
    <td>Improvement</td>
    <td>+42.5%</td>
    <td>+53.4%</td>
    <td>+73.4%</td>
    <td>+84.7%</td>
    <td>+63.5%</td>
  </tr>
</table>
</div>

*   Achieves state-of-the-art training for Vision-Language-Action models using PPO and GRPO.
*   Seamless integration with benchmarks like ManiSkill3 and LIBERO.

### Math Reasoning

<div align="center">
<table>
  <tr>
    <th colspan="5" style="text-align:center;"><strong>1.5B model results</strong></th>
  </tr>
  <tr>
    <th>Model</th>
    <th><a href="https://huggingface.co/datasets/RLinf/AIME24">AIME 24</a></th>
    <th><a href="https://huggingface.co/datasets/RLinf/AIME25">AIME 25</a></th>
    <th><a href="https://huggingface.co/datasets/RLinf/GPQA-diamond">GPQA-diamond</a></th>
    <th>Average</th>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">DeepSeek-R1-Distill-Qwen-1.5B (base model)</a></td>
    <td>28.33</td><td>24.90</td><td>27.45</td><td>26.89</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/zwhe99/DeepMath-1.5B"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">DeepMath-1.5B</a></td>
    <td>37.80</td><td>30.42</td><td>32.11</td><td>33.44</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">DeepScaleR-1.5B-Preview</a></td>
    <td>40.41</td><td>30.93</td><td>27.54</td><td>32.96</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/inclusionAI/AReaL-1.5B-Preview-Stage-3"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">AReaL-1.5B-Preview-Stage-3</a></td>
    <td>40.73</td><td>31.56</td><td>28.10</td><td>33.46</td>
  </tr>
  <tr>
    <td>AReaL-1.5B-retrain*</td>
    <td>44.42</td><td>34.27</td><td>33.81</td><td>37.50</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Nickyang/FastCuRL-1.5B-V3"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">FastCuRL-1.5B-V3</a></td>
    <td>43.65</td><td>32.49</td><td>35.00</td><td>37.05</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/RLinf/RLinf-math-1.5B"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;"><strong>RLinf-math-1.5B</strong></a></td>
    <td><strong>48.44</strong></td><td><strong>35.63</strong></td><td><strong>38.46</strong></td><td><strong>40.84</strong></td>
  </tr>
</table>
</div>

\* We retrain the model using the default settings for 600 steps.

<div align="center">
<table>
  <tr>
    <th colspan="5" style="text-align:center;"><strong>7B model results</strong></th>
  </tr>
  <tr>
    <th>Model</th>
    <th><a href="https://huggingface.co/datasets/RLinf/AIME24">AIME 24</a></th>
    <th><a href="https://huggingface.co/datasets/RLinf/AIME25">AIME 25</a></th>
    <th><a href="https://huggingface.co/datasets/RLinf/GPQA-diamond">GPQA-diamond</a></th>
    <th>Average</th>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">DeepSeek-R1-Distill-Qwen-7B (base model)</a></td>
    <td>54.90</td><td>40.20</td><td>45.48</td><td>46.86</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/inclusionAI/AReaL-boba-RL-7B"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">AReaL-boba-RL-7B</a></td>
    <td>61.66</td><td>49.38</td><td>46.93</td><td>52.66</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Skywork/Skywork-OR1-7B"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">Skywork-OR1-7B</a></td>
    <td>66.87</td><td>52.49</td><td>44.43</td><td>54.60</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/POLARIS-Project/Polaris-7B-Preview"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">Polaris-7B-Preview</a></td>
    <td><strong>68.55</strong></td><td>51.24</td><td>43.88</td><td>54.56</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/nvidia/AceMath-RL-Nemotron-7B"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">AceMath-RL-Nemotron-7B</a></td>
    <td>67.30</td><td><strong>55.00</strong></td><td>45.57</td><td>55.96</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/RLinf/RLinf-math-7B"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;"><strong>RLinf-math-7B</strong></a></td>
    <td>68.33</td><td>52.19</td><td><strong>48.18</strong></td><td><strong>56.23</strong></td>
  </tr>
</table>
</div>

*   Achieves state-of-the-art performance on math reasoning tasks across multiple benchmarks.

## Roadmap

*   System-Level Enhancements (Heterogeneous GPUs, Asynchronous Pipeline Execution, MoE, vLLM).
*   Application-Level Extensions (VLMs, deep searcher agents, multi-agent training, more embodied simulators, VLA models, world models, real-world RL).

## Getting Started

Comprehensive documentation is available [here](https://rlinf.readthedocs.io/en/latest/).

**Quickstart**

*   [Installation](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html)
*   [Quickstart 1: PPO Training of VLAs on Maniskill3](https://rlinf.readthedocs.io/en/latest/rst_source/start/vla.html)
*   [Quickstart 2: GRPO Training of LLMs on MATH](https://rlinf.readthedocs.io/en/latest/rst_source/start/llm.html)
*   [Multi-node Training](https://rlinf.readthedocs.io/en/latest/rst_source/start/distribute.html)
*   [Model Evaluation](https://rlinf.readthedocs.io/en/latest/rst_source/start/eval.html)

**Key Design**

*   [Unified User Interface Usage](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/user/index.html)
*   [Flexible Execution Modes](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/mode/index.html)
*   [Enable Automatic Scheduling](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/scheduler/index.html)
*   [Elastic Communication](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/communication/index.html)

**Example Gallery**

*   [Embodied Intelligence Vision-Language-Action Model training](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied.html)
*   [Math Reasoning Model Training](https://rlinf.readthedocs.io/en/latest/rst_source/examples/reasoning.html)

**Advanced Features**

*   [5D Parallelism Configuration for Megatron-LM](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/5D.html)
*   [LoRA Integration for efficient fine-tuning](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/lora.html)
*   [Switch between different versions of SGLang](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/version.html)
*   [Checkpoint Resume and Recovery Support](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/resume.html)

**Extending The Framework:**

*   [Adding new Environments](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_env.html)
*   [Adding new Models with FSDP+Huggingface backend](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_model_fsdp.html)
*   [Adding new Models with Megatron+SGLang backend](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_model_megatron.html)

**Blogs**

*   [Comparison with VeRL](https://rlinf.readthedocs.io/en/latest/rst_source/blog/compare_with_verl.html)

## Build Status

| Type              | Status |
| :---------------: | :----: |
| Reasoning RL-MATH | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml) |
| Embodied RL-VLA    | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml) |

## Contribution Guidelines

We welcome contributions!  See the [contribution guide](https://rlinf.readthedocs.io/en/latest/index.html#contribution-guidelines).

## Citation and Acknowledgement

If you use RLinf, please cite our paper:

```bibtex
@misc{yu2025rlinfflexibleefficientlargescale,
  title={RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation}, 
  author={Chao Yu and Yuanqing Wang and Zhen Guo and Hao Lin and Si Xu and Hongzhi Zang and Quanlu Zhang and Yongji Wu and Chunyang Zhu and Junhao Hu and Zixiao Huang and Mingjie Wei and Yuqing Xie and Ke Yang and Bo Dai and Zhexuan Xu and Xiangyuan Wang and Xu Fu and Zhihao Liu and Kang Chen and Weilin Liu and Gang Liu and Boxun Li and Jianlei Yang and Zhi Yang and Guohao Dai and Yu Wang},
  year={2025},
  eprint={2509.15965},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2509.15965}, 
}
```

If you use RL+VLA in RLinf, cite this empirical study:

```bibtex
@misc{liu2025rlbringvlageneralization,
  title={What Can RL Bring to VLA Generalization? An Empirical Study}, 
  author={Jijia Liu and Feng Gao and Bingwen Wei and Xinlei Chen and Qingmin Liao and Yi Wu and Chao Yu and Yu Wang},
  year={2025},
  eprint={2505.19789},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2505.19789}, 
}
```

**Acknowledgements**

RLinf leverages open-source ideas and tools. Thanks to the teams behind VeRL, AReaL, Megatron-LM, SGLang, and PyTorch FSDP. If we missed your project, please open an issue or pull request.

**Contact:**

We are looking for Postdocs, PhD/Master's students, and interns! Contact:

*   Chao Yu: zoeyuchao@gmail.com
*   Yu Wang: yu-wang@tsinghua.edu.cn