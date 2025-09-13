<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

# M3-Agent: Your AI Assistant with Long-Term Memory

**M3-Agent is a cutting-edge multimodal AI agent capable of seeing, listening, remembering, and reasoning, designed to understand and interact with the world like a human.**  Check out the original repo [here](https://github.com/ByteDance-Seed/m3-agent)

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

## Key Features:

*   **Multimodal Understanding:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Builds and updates an entity-centric multimodal memory.
*   **Semantic Memory:** Develops world knowledge over time.
*   **Iterative Reasoning:** Performs multi-turn reasoning to accomplish tasks.
*   **Enhanced Performance:** Outperforms baseline models in long-video question answering.

## M3-Bench: A New Benchmark for Multimodal Agents

M3-Agent's performance is evaluated on M3-Bench, a novel long-video question-answering benchmark. The benchmark includes:

*   **M3-Bench-robot:** 100 real-world videos from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos covering diverse scenarios.

### M3-Bench Examples:

[link1](https://www.youtube.com/watch?v=7W0gRqCRMZQ), [link2](https://www.youtube.com/watch?v=Efk3K4epEzg), [link3](https://www.youtube.com/watch?v=6Unxpxy-Ct4)

### Statistics of M3-Bench Benchmark:

![architecture](figs/m3-bench-statistic.png)

## Run M3-Agent Locally

Follow these steps to get started:

1.  **Set up environment:**
    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```
2.  **Configure API:** Add your API config in `configs/api_config.json`.
3.  **Run Memorization:** Generate memory graphs.
    *   Download necessary intermediate outputs and memory graphs from Hugging Face or generate them yourself.
    *   If generating, follow the cutting video, prepare data, generate intermediate outputs, and generate memory graphs steps in the original README.
4.  **Run Control:** Perform question answering and evaluation.  Download M3-Agent-Control from Hugging Face.
    ```bash
    python m3_agent/control.py \
        --data_file data/annotations/robot.json
    ```
5.  **Memory Graph Visualization:** Visualize the memory graph:
    ```bash
    python visualization.py \
        --mem_path data/memory_graphs/robot/bedroom_01.pkl \
        --clip_id 1
    ```
6.  **Training:** Memorization: [https://github.com/hyc2026/sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker) and Control: [https://github.com/hyc2026/M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

## Citation

```BibTeX
@misc{long2025seeing,
      title={Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory}, 
      author={Lin Long, Yichen He, Wentao Ye, Yiyuan Pan, Yuan Lin, Hang Li, Junbo Zhao, Wei Li},
      year={2025},
      eprint={2508.09736},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```