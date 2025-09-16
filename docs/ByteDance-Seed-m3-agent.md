<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: The Multimodal Agent Revolutionizing Long-Term Memory and Reasoning</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

M3-Agent is a cutting-edge multimodal agent that leverages long-term memory to understand, remember, and reason about complex visual and auditory information, enabling more human-like intelligence. **[Visit the original repository](https://github.com/ByteDance-Seed/m3-agent) for more details.**

## Key Features

*   **Multimodal Understanding:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Builds and updates both episodic and semantic memory.
*   **Entity-Centric Memory:** Organizes memory in a multimodal graph for deeper understanding.
*   **Iterative Reasoning:** Performs multi-turn reasoning and retrieves relevant information for task completion.
*   **Enhanced Performance:** Outperforms existing baselines in long video question answering.

## M3-Bench: A Benchmark for Multimodal Reasoning

M3-Bench is a novel benchmark designed to evaluate multimodal agents' long-term memory and reasoning capabilities. It features:

*   **M3-Bench-robot:** 100 real-world videos from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos covering diverse scenarios.
*   **Question-Answer Pairs:** Designed to test human understanding, general knowledge extraction, and cross-modal reasoning.

**[See examples here:](https://www.youtube.com/watch?v=7W0gRqCRMZQ), [here:](https://www.youtube.com/watch?v=Efk3K4epEzg), and [here:](https://www.youtube.com/watch?v=6Unxpxy-Ct4)**

![architecture](figs/m3-bench-example.png)

### Data and Intermediate Outputs

*   **M3-Bench-robot Videos:** Downloadable from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot).
*   **M3-Bench-web Videos:** Found via video\_url in `data/annotations/web.json`.
*   **Intermediate Outputs:** Available for download on [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) or generated from videos (see instructions).
*   **Memory Graphs:** Available for download on [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs) or generated from videos (see instructions).

![architecture](figs/m3-bench-statistic.png)

## M3-Agent Architecture

![architecture](figs/m3-agent.png)

M3-Agent's architecture consists of two parallel processes:

*   **Memorization:** Processes video and audio streams to create episodic and semantic memory.
*   **Control:** Executes instructions through iterative reasoning and retrieval from long-term memory.

## Experimental Results

![architecture](figs/exp_result.png)

M3-Agent demonstrates superior performance on M3-Bench and VideoMME-long benchmarks.

## Run Locally

Instructions for setting up your environment and running the M3-Agent are provided.

### Memorization

1.  **Setup Environment:** Follow the setup instructions provided in `setup.sh` and install necessary packages.
2.  **Cut Video:** This step requires `ffmpeg` to split videos into 30-second segments.  A bash script is provided.
3.  **Prepare Data:** Create a JSONL file (`data/data.jsonl`) with video metadata.
4.  **Generate Intermediate Outputs:** Run `m3_agent/memorization_intermediate_outputs.py` to generate intermediate outputs using face detection and speaker diarization.  Requires downloading a pre-trained audio embedding model and speakerlab.
5.  **Generate Memory Graphs:** Run `m3_agent/memorization_memory_graphs.py` using the M3-Agent-Memorization model.
6.  **Memory Graph Visualization:** Use `visualization.py` to visualize memory graphs.

### Control

1.  **Setup Environment:** Follow the setup instructions and install the required packages including `transformers`, `vllm`, and `numpy`.
2.  **Question Answering and Evaluation:** Run `m3_agent/control.py` to generate answers using the M3-Agent-Control model and evaluate them with GPT-4o.

### Other Models

Instructions for using other models by changing API calls and prompts.

## Training

*   Memorization: [https://github.com/hyc2026/sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
*   Control: [https://github.com/hyc2026/M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

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