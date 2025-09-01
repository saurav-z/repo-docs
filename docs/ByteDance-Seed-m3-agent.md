<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: Build a human-like agent that *Sees, Listens, Remembers, and Reasons*.</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

**[Explore the original repository](https://github.com/ByteDance-Seed/m3-agent) for more details.**

## Overview

M3-Agent is a cutting-edge multimodal agent framework designed to emulate human-like cognitive abilities. It integrates vision, audio, and long-term memory to understand and interact with its environment. Unlike traditional agents, M3-Agent builds both episodic and semantic memory, enabling it to accumulate knowledge and reason effectively over time. M3-Agent excels at tasks requiring deep understanding and multi-turn reasoning, as demonstrated by its superior performance on the M3-Bench benchmark.

## Key Features

*   **Multimodal Understanding:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Develops and utilizes both episodic and semantic memory.
*   **Entity-Centric Memory:** Organizes memory in a structured, multimodal graph format for consistent understanding.
*   **Iterative Reasoning:** Employs multi-turn reasoning and retrieves relevant information to accomplish tasks.
*   **Superior Performance:** Outperforms state-of-the-art models on the M3-Bench benchmark.

## M3-Bench: A Benchmark for Multimodal Agents

M3-Bench is a novel benchmark designed to evaluate the effectiveness of memory and reasoning in multimodal agents. The dataset includes:

*   **M3-Bench-robot:** 100 videos from a robot's perspective in realistic scenarios.
*   **M3-Bench-web:** 920 web-sourced videos covering diverse content and situations.
*   Question-answer pairs designed to test key capabilities such as understanding, knowledge extraction, and cross-modal reasoning.

### Example Videos
[Example 1](https://www.youtube.com/watch?v=7W0gRqCRMZQ), [Example 2](https://www.youtube.com/watch?v=Efk3K4epEzg), [Example 3](https://www.youtube.com/watch?v=6Unxpxy-Ct4)

![M3-Bench example](figs/m3-bench-example.png)

![M3-Bench statistics](figs/m3-bench-statistic.png)

### Download Data

1.  **M3-Bench-robot:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)
2.  **M3-Bench-web:** Download from video\_url in `data/annotations/web.json`

### Intermediate Outputs
Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) or generate them from videos (see below).

### Memory Graphs
Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs) or generate them from videos (see below).

## M3-Agent Architecture

![M3-Agent Architecture](figs/m3-agent.png)

The M3-Agent architecture consists of two parallel processes:

*   **Memorization:** Processes video and audio streams to create episodic and semantic memory.
*   **Control:** Executes instructions through iterative reasoning and memory retrieval.

## Experimental Results

![Experiment Results](figs/exp_result.png)

M3-Agent demonstrates superior performance on M3-Bench-robot, M3-Bench-web, and VideoMME-long, showcasing its advanced capabilities in multimodal reasoning and long-term memory.

## Run Locally

### Memorization

1.  **Setup Environment:**  Run `bash setup.sh` and install dependencies:

    ```bash
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```

2.  **Cut Video:**  Use the provided script to segment videos into 30-second clips (example below):

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

3.  **Prepare Data:** Create a JSONL file (`data/data.jsonl`) with video information:

    ```json
    {"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
    ```

4.  **Generate Intermediate Outputs:**  Run the following script.  Requires Face Detection and Speaker Diarization tools, along with audio embedding model ([pretrained\_eres2netv2.ckpt](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt)) and [speakerlab](https://github.com/modelscope/3D-Speaker/tree/main/speakerlab).

    ```bash
    python m3_agent/memorization_intermediate_outputs.py --data_file data/data.jsonl
    ```

5.  **Generate Memory Graphs:**  Use the M3-Agent-Memorization model.  Download the model from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot) and then run:

    ```bash
    python m3_agent/memorization_memory_graphs.py --data_file data/data.jsonl
    ```

6.  **Memory Graph Visualization:**

    ```bash
    python visualization.py --mem_path data/memory_graphs/robot/bedroom_01.pkl --clip_id 1
    ```

### Control

1.  **Setup Environment:** Install dependencies:

    ```bash
    bash setup.sh
    pip install transformers==4.51.0
    pip install vllm==0.8.4
    pip install numpy==1.26.4
    ```

2.  **Question Answering and Evaluation:** Use the M3-Agent-Control model. Download the model from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/blob/main/videos/robot) and run:

    ```bash
    python m3_agent/control.py --data_file data/annotations/robot.json
    ```

### Other Models

You can adapt the prompts to other models for memory generation and question answering. See `mmagent.prompts` for examples.

## Training

1.  Memorization:  [SFT-Qwen2.5-Omni-Thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
2.  Control:  [M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

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