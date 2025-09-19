<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: Revolutionizing Multimodal Agents with Long-Term Memory</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

**M3-Agent is a cutting-edge multimodal agent that learns, remembers, and reasons, enabling it to understand and interact with its environment more like humans.**

[View the original repository](https://github.com/ByteDance-Seed/m3-agent)

## Key Features

*   **Long-Term Memory:** M3-Agent builds and updates a multimodal long-term memory, similar to human memory, for enhanced understanding.
*   **Multimodal Processing:** Processes real-time visual and auditory inputs for comprehensive environmental awareness.
*   **Entity-Centric Memory:** Organizes memory in a structured, entity-centric format for a deeper understanding of the environment.
*   **M3-Bench Benchmark:** Features M3-Bench, a new benchmark to evaluate memory effectiveness in multimodal agents.
*   **Superior Performance:** Outperforms existing models on benchmarks like M3-Bench and VideoMME-long.
*   **Personal Assistant Demo:** Watch the M3-Agent in action!
    [![Watch the video](figs/demo.png)](https://www.youtube.com/watch?v=XUx31cBanfo)
    *Video on [Bilibili](https://www.bilibili.com/video/BV1h9YpznEx9/)*

## M3-Bench: Evaluating Long-Term Memory and Reasoning

M3-Bench is a dedicated benchmark designed to assess the capabilities of multimodal agents in reasoning over long-term memory. It comprises:

*   **M3-Bench-robot:** 100 real-world videos from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos covering diverse scenarios.

*Examples from M3-Bench.*

![architecture](figs/m3-bench-example.png)

**Statistical Overview of M3-Bench:**

![architecture](figs/m3-bench-statistic.png)

### Accessing M3-Bench

1.  **M3-Bench-robot Videos:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot).
2.  **M3-Bench-web Videos:** Access via the `video_url` in `data/annotations/web.json`.

### Intermediate Outputs and Memory Graphs

*   **Download (Optional):** Pre-processed outputs and memory graphs are available on [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) and [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs), respectively.
*   **Generate (Optional):** Instructions for generating intermediate outputs and memory graphs are provided in the "Run Locally" section.

## M3-Agent Architecture

The M3-Agent employs a two-process architecture for efficient operation:

![architecture](figs/m3-agent.png)

*   **Memorization:** Processes video and audio streams to create episodic and semantic memory.
*   **Control:** Executes instructions through iterative reasoning and retrieval from long-term memory.

## Experimental Results

*M3-Agent performance on M3-Bench and VideoMME-long.*

![architecture](figs/exp_result.png)

## Run M3-Agent Locally

*Before running, configure API settings in `configs/api_config.json`.*

### Memorization

*Steps to generate memory graphs. Skip these if you've downloaded pre-processed outputs.*

1.  **Environment Setup:** Execute `bash setup.sh` and install dependencies.
2.  **Video Segmentation:** Cut videos into 30-second segments using the provided script.
3.  **Data Preparation:** Create a JSONL file (`data/data.jsonl`) with video information.
4.  **Generate Intermediate Outputs:** Run `m3_agent/memorization_intermediate_outputs.py`. Requires speakerlab and a pretrained audio embedding model.
5.  **Generate Memory Graphs:** Execute `m3_agent/memorization_memory_graphs.py` using the M3-Agent-Memorization model.
6.  **Visualize Memory Graphs:** Use `visualization.py` to view memory graphs.

### Control

1.  **Environment Setup:** Execute `bash setup.sh` and install the listed dependencies.
2.  **Question Answering and Evaluation:** Utilize the M3-Agent-Control model for question answering and evaluation.

### Other Models

Modify model inference to use different API calls and prompts for alternative models.

*   **Prompts:**
    *   Memorization: Gemini/GPT-4o (`mmagent.prompts.prompt_generate_captions_with_ids`) / Qwen2.5-Omni-7B (`mmagent.prompts.prompt_generate_full_memory`).
    *   Control: GPT-4o (`mmagent.prompts.prompt_answer_with_retrieval_final`).

## Training Details

*   **Memorization:** Refer to [sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker).
*   **Control:** Refer to [M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training).

## Citation

Cite our work as:

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