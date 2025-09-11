<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: A Multimodal Agent with Long-Term Memory</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

**M3-Agent is a groundbreaking multimodal AI agent that learns and reasons like a human, leveraging long-term memory for enhanced understanding and task completion.**  For more details, visit the [original M3-Agent repository](https://github.com/ByteDance-Seed/m3-agent).

## Key Features

*   **Multimodal Perception:** Processes visual and auditory inputs in real-time.
*   **Long-Term Memory:** Builds and updates both episodic and semantic memory for comprehensive knowledge.
*   **Entity-Centric Organization:**  Stores information in a multimodal graph format for deeper environmental understanding.
*   **Iterative Reasoning:** Autonomously reasons and retrieves relevant information from memory to complete tasks.
*   **M3-Bench Benchmark:** Includes M3-Bench, a new long-video question answering dataset to evaluate the effectiveness of memory-based reasoning.
*   **Strong Performance:** Outperforms leading baselines on the M3-Bench and VideoMME-long benchmarks.

## Overview

M3-Agent is designed to understand and interact with its environment in a more human-like way. The agent's architecture consists of two core processes: memorization and control. During memorization, the agent processes visual and audio streams to create and update its memory. During control, it utilizes this memory to answer questions and complete tasks.

![architecture](figs/m3-agent.png)

## M3-Bench: Evaluating Memory and Reasoning

M3-Bench is a crucial benchmark for assessing the performance of multimodal agents like M3-Agent. It includes long-form videos and associated question-answer pairs.

*   **M3-Bench-robot:** 100 videos recorded from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos across diverse scenarios.

![architecture](figs/m3-bench-example.png)
Examples from M3-Bench. M3-Bench-robot features long videos from realistic robotic work scenarios, while M3-Bench-web expands the video diversity to support broader evaluation. The question-answering tasks are designed to assess a multimodal agent’s ability to construct consistent and reliable long-term memory, as well as to reason effectively over that memory.

![architecture](figs/m3-bench-statistic.png)
Statistical overview of M3-Bench benchmark. Each question may correspond to multiple question types.

### Dataset Resources

*   **M3-Bench-robot Videos:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)
*   **M3-Bench-web Videos:** Download from video_url in `data/annotations/web.json`
*   **Intermediate Outputs:** Optional - Download pre-processed outputs from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) or generate them (see instructions below).
*   **Memory Graphs:** Optional - Download pre-generated memory graphs from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs) or generate them (see instructions below).

## Run M3-Agent Locally

> Before running, add api config in `configs/api_config.json`

### 1. Memorization

These instructions describe how to generate memory graphs for videos. It's only required if you haven't downloaded the pre-processed  *intermediate_outputs* and *memory_graphs* from Hugging Face, or if you wish to process custom videos.

1.  **Set up the environment:**

    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```

2.  **Cut Video**

    Cut the video into 30 second segments.

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

3.  **Prepare Data**

    Create a JSONL file (`data/data.jsonl`) with a single video per line.

    ```json
    {"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
    ```

4.  **Generate Intermediate Outputs**

    **(Requires Face Detection and Speaker Diarization)**

    *   **Important:** If you are using M3-Bench and have downloaded the `intermediate_outputs` from Hugging Face, you can skip this step.
    *   **Dependencies:**
        *   Download the audio embedding model and save it into `models\` from [pretrained_eres2netv2.ckpt](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt)
        *   Download `speakerlab` from [github.com/modelscope/3D-Speaker/tree/main/speakerlab](https://github.com/modelscope/3D-Speaker/tree/main/speakerlab)

        ```
        m3-agent
        ├── models
        │   └── pretrained_eres2netv2.ckpt
        └── speakerlab
        ```

    ```bash
    python m3_agent/memorization_intermediate_outputs.py \
       --data_file data/data.jsonl
    ```

5.  **Generate Memory Graphs**

    **(Requires the M3-Agent-Memorization model)**

    *   Download the M3-Agent-Memorization model from [Hugging Face](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)

    ```bash
    python m3_agent/memorization_memory_graphs.py \
       --data_file data/data.jsonl
    ```

6.  **Memory Graph Visualization**

    ```bash
    python visualization.py \
       --mem_path data/memory_graphs/robot/bedroom_01.pkl \
       --clip_id 1
    ```

### 2. Control

1.  **Set up the environment:**

    ```bash
    bash setup.sh
    pip install transformers==4.51.0
    pip install vllm==0.8.4
    pip install numpy==1.26.4
    ```

2.  **Question Answering and Evaluation**

    **(Requires the M3-Agent-Control model and GPT-4o for evaluation)**

    *   Download the M3-Agent-Control model from [Hugging Face](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)

    ```bash
    python m3_agent/control.py \
       --data_file data/annotations/robot.json
    ```

### 3. Using Other Models

To utilize different models for memory generation or question answering, simply adapt the API calls with the appropriate prompts.

**Prompts:**

1.  **Memorization**
    *   Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
    *   Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`

2.  **Control**
    *   GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Experimental Results

![architecture](figs/exp_result.png)

Results on M3-Bench-robot, M3-Bench-web, and VideoMME-long.

## Training

Resources for training the M3-Agent:

1.  Memorization: [github.com/hyc2026/sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
2.  Control: [github.com/hyc2026/M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

## Citation

If you use M3-Agent in your research, please cite the following paper:

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