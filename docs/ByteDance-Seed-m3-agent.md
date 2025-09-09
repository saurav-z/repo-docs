<!-- Image for visual appeal and branding -->
<div align="center">
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

# M3-Agent: A Multimodal Agent with Long-Term Memory

> **M3-Agent empowers AI to understand and interact with the world more like humans, remembering past experiences to reason and complete tasks.**

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

**[View the original repository on GitHub](https://github.com/ByteDance-Seed/m3-agent)**

## Key Features

*   **Multimodal Understanding:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Builds and updates episodic and semantic memory for robust understanding.
*   **Entity-Centric Memory:** Organizes memory in a multimodal graph, enhancing consistency.
*   **Iterative Reasoning:** Performs multi-turn reasoning and information retrieval for task completion.
*   **M3-Bench Benchmark:** Utilizes a novel benchmark for evaluating multimodal agent performance.

## Overview

M3-Agent is a cutting-edge multimodal agent framework designed to emulate human-like cognitive abilities. This innovative system is capable of seeing, hearing, remembering, and reasoning, enabling it to understand and interact with the world in a more natural and effective way. It leverages long-term memory to accumulate world knowledge and enhance decision-making.

## M3-Bench: Evaluating Long-Term Memory

M3-Agent's capabilities are assessed using the M3-Bench, a specialized long-video question-answering benchmark. This dataset is designed to evaluate the effectiveness of long-term memory and memory-based reasoning in multimodal agents.

*   **M3-Bench-robot:** Features 100 real-world videos recorded from a robot's perspective.
*   **M3-Bench-web:** Includes 920 web-sourced videos across a variety of scenarios.

![M3-Bench Example](figs/m3-bench-example.png)
*Examples from M3-Bench. M3-Bench-robot features long videos from realistic robotic work scenarios, while M3-Bench-web expands the video diversity to support broader evaluation. The question-answering tasks are designed to assess a multimodal agent’s ability to construct consistent and reliable long-term memory, as well as to reason effectively over that memory.*

![M3-Bench Statistics](figs/m3-bench-statistic.png)
*Statistical overview of M3-Bench benchmark. Each question may correspond to multiple question types.*

### Dataset Resources

*   **M3-Bench-robot:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)
*   **M3-Bench-web:** Access video URLs from `data/annotations/web.json`
*   **Intermediate Outputs:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) or generate them yourself.
*   **Memory Graphs:** Download pre-processed memory graphs from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs) or generate them from video data.

## M3-Agent Architecture

![M3-Agent Architecture](figs/m3-agent.png)
*Architecture of M3-Agent. The system consists of two parallel processes: memorization and control. During memorization, M3-Agent processes video and audio streams online to generate episodic and semantic memory. During control, it executes instructions by iteratively thinking and retrieving from long-term memory. The long-term memory is structured as a multimodal graph.*

M3-Agent is designed with two primary processes: memorization and control. The memorization process involves generating episodic and semantic memory from video and audio streams. The control process executes instructions through iterative reasoning and retrieval from the agent's long-term, multimodal graph-structured memory.

## Experimental Results

![Experimental Results](figs/exp_result.png)
*Results on M3-Bench-robot, M3-Bench-web, and VideoMME-long.*

The experimental results on M3-Bench and VideoMME-long demonstrate M3-Agent's superior performance compared to existing methods, showcasing its effectiveness in long-term memory and reasoning tasks.

## Run M3-Agent Locally

### Prerequisites

*   Ensure you have the required API configurations in `configs/api_config.json`.

### Steps

1.  **Set up the Environment:**

    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```

2.  **Cut Videos (if not using M3-Bench):**

    ```bash
    #!/bin/bash

    video="robot/bedroom_01"
    input="data/videos/$video.mp4"
    mkdir -p "data/clips/$video"
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$input")
    duration_seconds=$(echo "$duration" | awk '{print int($1)}')

    segments=$((duration_seconds / 30 + 1))
    for ((i=0; i<segments; i++)); do
        start=$((i * 30))
        end=$(((i + 1) * 30))
        output="data/clips/$video/$i.mp4"
        ffmpeg -ss $start -i "$input" -t 30 -c copy "${output}"
    done
    ```

3.  **Prepare Data:**

    Create a `data/data.jsonl` file with the video details:

    ```json
    {"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
    ```

4.  **Generate Intermediate Outputs:**

    Requires Face Detection and Speaker Diarization tools. Download the audio embedding model and speakerlab.

    ```bash
    python m3_agent/memorization_intermediate_outputs.py \
       --data_file data/data.jsonl
    ```

5.  **Generate Memory Graphs:**

    Requires the M3-Agent-Memorization model. Download the model from Hugging Face.

    ```bash
    python m3_agent/memorization_memory_graphs.py \
       --data_file data/data.jsonl
    ```

6.  **Memory Graph Visualization:**

    ```bash
    python visualization.py \
       --mem_path data/memory_graphs/robot/bedroom_01.pkl \
       --clip_id 1
    ```

### Control

1.  **Set up the Environment:**

    ```bash
    bash setup.sh
    pip install transformers==4.51.0
    pip install vllm==0.8.4
    pip install numpy==1.26.4
    ```

2.  **Question Answering and Evaluation:**

    Requires the M3-Agent-Control model. Download the model from Hugging Face.

    ```bash
    python m3_agent/control.py \
       --data_file data/annotations/robot.json
    ```

### Model Integration

To incorporate other models, modify the model inference calls and utilize the corresponding prompts for memorization and control:

1.  **Memorization:**
    *   Gemini/GPT-4o: Use `mmagent.prompts.prompt_generate_captions_with_ids`
    *   Qwen2.5-Omni-7B: Use `mmagent.prompts.prompt_generate_full_memory`

2.  **Control:**
    *   GPT-4o: Use `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training Resources

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