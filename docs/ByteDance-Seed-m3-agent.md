<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

# M3-Agent: Building Human-Like Long-Term Memory in Multimodal Agents

**M3-Agent** introduces a novel multimodal agent capable of "seeing, listening, remembering, and reasoning" to perform complex tasks.  Learn more and explore the code on the original repo: [https://github.com/ByteDance-Seed/m3-agent](https://github.com/ByteDance-Seed/m3-agent).

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

## Key Features

*   **Multimodal Processing:** M3-Agent integrates visual and auditory inputs for a comprehensive understanding of its environment.
*   **Long-Term Memory:** The agent builds and updates long-term memory, including both episodic and semantic knowledge.
*   **Entity-Centric Memory:** Information is stored in an entity-centric, multimodal format, facilitating a deeper and more consistent understanding.
*   **Iterative Reasoning:** M3-Agent performs multi-turn, iterative reasoning, retrieving relevant information from memory to accomplish tasks.
*   **M3-Bench Benchmark:**  A new benchmark is introduced to evaluate memory effectiveness and reasoning in multimodal agents.

## M3-Bench: Evaluating Long-Term Memory and Reasoning

M3-Bench is a comprehensive long-video question-answering dataset designed to assess the capabilities of multimodal agents in reasoning over extended periods. It includes two subsets:

*   **M3-Bench-robot:** 100 real-world videos from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos across diverse scenarios.

![m3-bench-example](figs/m3-bench-example.png)
*Examples from M3-Bench.*

![m3-bench-statistic](figs/m3-bench-statistic.png)
*Statistical overview of M3-Bench benchmark.*

### Dataset Access

*   **M3-Bench-robot Videos:**  Available on [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)
*   **M3-Bench-web Videos:**  Download URLs are in `data/annotations/web.json`.
*   **Intermediate Outputs** are available for download on [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs)
*   **Memory Graphs** are available for download on [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs)

## M3-Agent Architecture

![m3-agent](figs/m3-agent.png)

M3-Agent's architecture comprises two parallel processes: memorization and control. The memorization process generates episodic and semantic memory from video and audio streams, while the control process executes instructions through iterative reasoning and retrieval from the long-term memory.

## Experimental Results

![exp_result](figs/exp_result.png)

M3-Agent achieves superior performance on M3-Bench and VideoMME-long compared to baseline models.

## Run Locally

Follow these steps to run M3-Agent:

### Memorization

1.  **Setup Environment:** Run `bash setup.sh` and install the required Python packages (see original README for details).
2.  **Cut Video (Optional):** Use the provided `cut_video.sh` script to segment videos into 30-second clips.
3.  **Prepare Data:** Create a `data/data.jsonl` file with video information.
4.  **Generate Intermediate Outputs (Optional):**  Run `python m3_agent/memorization_intermediate_outputs.py`. This step requires a face detection model and speaker diarization tools.
5.  **Generate Memory Graphs:**  Run `python m3_agent/memorization_memory_graphs.py`.
6.  **Memory Graph Visualization:** Run `python visualization.py`.

### Control

1.  **Setup Environment:** Install dependencies.
2.  **Question Answering and Evaluation:** Run `python m3_agent/control.py`.

### Other Models

Modify API calls and prompts to use other models for memory generation and question answering.

## Training

*   **Memorization:**  See [https://github.com/hyc2026/sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
*   **Control:** See [https://github.com/hyc2026/M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

## Citation

```bibtex
@misc{long2025seeing,
      title={Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory}, 
      author={Lin Long, Yichen He, Wentao Ye, Yiyuan Pan, Yuan Lin, Hang Li, Junbo Zhao, Wei Li},
      year={2025},
      eprint={2508.09736},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}