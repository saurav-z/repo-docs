<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: A Multimodal Agent with Human-Like Long-Term Memory</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

**Unlock advanced multimodal understanding with M3-Agent, an innovative framework that mimics human-like long-term memory and reasoning abilities.** Explore the original repository here: [ByteDance-Seed/m3-agent](https://github.com/ByteDance-Seed/m3-agent)

## Key Features:

*   **Multimodal Perception:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Builds and updates both episodic and semantic memory for comprehensive understanding.
*   **Entity-Centric Memory:** Organizes information in a structured, multimodal graph format for deeper contextual understanding.
*   **Iterative Reasoning:** Employs multi-turn reasoning and memory retrieval to accomplish tasks.
*   **M3-Bench Benchmark:** Includes M3-Bench-robot and M3-Bench-web, specifically designed for evaluating long-term memory and reasoning in multimodal agents.

## What is M3-Agent?

M3-Agent is a cutting-edge multimodal agent designed to perceive, understand, and reason about its environment in a way that mirrors human cognitive processes. By integrating visual and auditory inputs, the agent constructs a rich, entity-centric, long-term memory. This memory is crucial for the agent's ability to answer questions, solve problems, and interact effectively within complex scenarios. The project introduces M3-Bench, a benchmark designed to assess memory effectiveness and reasoning capabilities in multimodal agents.

## M3-Bench Dataset

M3-Bench is a groundbreaking long-video question-answering dataset tailored to evaluate multimodal agents' reasoning abilities over extended periods. It comprises:

*   **M3-Bench-robot:** 100 real-world videos captured from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos offering diverse scenarios.

The dataset facilitates rigorous testing of essential capabilities such as:

*   Human understanding
*   General knowledge extraction
*   Cross-modal reasoning

**Example Videos:**
[Link1](https://www.youtube.com/watch?v=7W0gRqCRMZQ), [Link2](https://www.youtube.com/watch?v=Efk3K4epEzg), [Link3](https://www.youtube.com/watch?v=6Unxpxy-Ct4)

## M3-Agent Architecture

The M3-Agent architecture is designed with two parallel processes: memorization and control.

*   **Memorization:** Continuously processes video and audio streams to create episodic and semantic memory.
*   **Control:** Executes instructions by iteratively reasoning and retrieving information from its long-term, multimodal graph-structured memory.

![architecture](figs/m3-agent.png)

## Experimental Results

M3-Agent demonstrates superior performance on M3-Bench and VideoMME-long compared to baseline models.

![architecture](figs/exp_result.png)

## Run M3-Agent Locally

Before starting, ensure you have configured the API settings in `configs/api_config.json`.

### 1. Memorization

Generate memory graphs for each video. The results are stored in `data/memory_graphs`.

*   Follow these steps only if you haven't downloaded *intermediate_outputs* and *memory_graphs* from huggingface or want to process videos beyond M3-Bench.

    1.  **Set Up Environment:**

        ```bash
        bash setup.sh
        pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
        pip install qwen-omni-utils==0.0.4
        ```

    2.  **Cut Video:**

        Cut videos into 30-second segments:

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

        Create a `data.jsonl` file with video information:

        ```json
        {"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
        ```

    4.  **Generate Intermediate Outputs:**

        Run the script to generate intermediate outputs using face detection and speaker diarization. Download models from specified links.

        ```bash
        python m3_agent/memorization_intermediate_outputs.py \
           --data_file data/data.jsonl
        ```

    5.  **Generate Memory Graphs:**

        Use the M3-Agent-Memorization model to produce memory graphs.

        ```bash
        python m3_agent/memorization_memory_graphs.py \
           --data_file data/data.jsonl
        ```

    6.  **Memory Graph Visualization:**

        Visualize the memory graph:

        ```bash
        python visualization.py \
           --mem_path data/memory_graphs/robot/bedroom_01.pkl \
           --clip_id 1
        ```

### 2. Control

1.  **Set Up Environment:**

    ```bash
    bash setup.sh
    pip install transformers==4.51.0
    pip install vllm==0.8.4
    pip install numpy==1.26.4
    ```

2.  **Question Answering and Evaluation:**

    Use the M3-Agent-Control model to generate answers and GPT-4o for evaluation.

    ```bash
    python m3_agent/control.py \
       --data_file data/annotations/robot.json
    ```

### Using Other Models

Easily adapt other models by modifying model inference to API calls and adjusting the prompts.

**Prompts:**

1.  Memorization
    *   Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
    *   Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`

2.  Control
    *   GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training

*   **Memorization:** [sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
*   **Control:** [M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

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