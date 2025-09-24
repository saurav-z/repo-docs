<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: A Multimodal Agent with Human-Like Long-Term Memory</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

**M3-Agent empowers AI with long-term memory and advanced reasoning capabilities, enabling it to understand and interact with the world more like humans.**

## Key Features:

*   **Multimodal Input Processing:**  Processes real-time visual and auditory data.
*   **Long-Term Memory:** Builds and updates both episodic and semantic memory.
*   **Entity-Centric Memory:** Organizes memory in a multimodal graph format for a deeper understanding.
*   **Iterative Reasoning:**  Performs multi-turn reasoning and retrieves relevant information to accomplish tasks.
*   **M3-Bench Benchmark:** Introduces a new benchmark for evaluating multimodal agents, including M3-Bench-robot and M3-Bench-web.
*   **Superior Performance:** Outperforms strong baselines (Gemini-1.5-pro and GPT-4o) on the M3-Bench benchmark.

## Overview

M3-Agent is a groundbreaking multimodal agent framework designed to mimic human-like cognitive abilities. By integrating long-term memory, M3-Agent can understand, remember, and reason about its environment in a more consistent and comprehensive manner. This is achieved through the integration of real-time visual and auditory input into a dynamic memory system. The agent's memory is structured around entities and organized in a multimodal format, fostering deeper understanding. The system is built on two parallel processes: memorization and control. During memorization, M3-Agent processes video and audio streams online to generate episodic and semantic memory. During control, it executes instructions by iteratively thinking and retrieving from long-term memory.

### M3-Bench: Evaluating Long-Term Memory and Reasoning

M3-Bench is a key component of the M3-Agent project. This long-video question-answering benchmark evaluates a multimodal agent's capacity for reasoning over long-term memory. It consists of two distinct subsets:

*   **M3-Bench-robot:** Features 100 real-world videos recorded from a robot's perspective.
*   **M3-Bench-web:** Includes 920 web-sourced videos spanning various scenarios.

### Example Videos
*   [M3-Bench-robot Example 1](https://www.youtube.com/watch?v=7W0gRqCRMZQ)
*   [M3-Bench-robot Example 2](https://www.youtube.com/watch?v=Efk3K4epEzg)
*   [M3-Bench-web Example](https://www.youtube.com/watch?v=6Unxpxy-Ct4)

## Run Locally

This section provides instructions on how to set up and run M3-Agent locally.

### Prerequisites

*   Add your API config in `configs/api_config.json`.
*   Follow the below setup and install instructions

### Installation

```bash
bash setup.sh
pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
pip install qwen-omni-utils==0.0.4
pip install transformers==4.51.0
pip install vllm==0.8.4
pip install numpy==1.26.4
```

### Memorization

**Steps required only if you haven't downloaded *intermediate_outputs* and *memory_graphs* from huggingface:**

1.  **Cut Video:** Convert videos into 30-second segments.
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
2.  **Prepare Data:** Create a JSONL file (`data/data.jsonl`) with video information:
    ```json
    {"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
    ```
3.  **Generate Intermediate Outputs:**  Use face detection and speaker diarization tools.
    *   Download audio embedding model and save into `models\` from [pretrained_eres2netv2.ckpt](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt)
    *   Download [speakerlab](https://github.com/modelscope/3D-Speaker/tree/main/speakerlab)
    ```bash
    python m3_agent/memorization_intermediate_outputs.py \
       --data_file data/data.jsonl
    ```
4.  **Generate Memory Graphs:**  Use the M3-Agent-Memorization model.
    *   Download M3-Agent-Memorization from [huggingface](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
    ```bash
    python m3_agent/memorization_memory_graphs.py \
       --data_file data/data.jsonl
    ```
5.  **Memory Graph Visualization:**
    ```bash
    python visualization.py \
       --mem_path data/memory_graphs/robot/bedroom_01.pkl \
       --clip_id 1
    ```

### Control

1.  **Question Answering and Evaluation:** Uses the M3-Agent-Control model and GPT-4o for answer generation and evaluation.
    *   Download M3-Agent-Control from [huggingface](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
    ```bash
    python m3_agent/control.py \
       --data_file data/annotations/robot.json
    ```

### Other Models

*   Customize the model inference to API calling for alternative models.

## Training

### Memorization: [SFT-Qwen2.5-Omni-Thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
### Control: [M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

## Citation

For academic use, please cite the following paper:

```bibtex
@misc{long2025seeing,
      title={Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory}, 
      author={Lin Long, Yichen He, Wentao Ye, Yiyuan Pan, Yuan Lin, Hang Li, Junbo Zhao, Wei Li},
      year={2025},
      eprint={2508.09736},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

**[Visit the M3-Agent Repository](https://github.com/ByteDance-Seed/m3-agent) for more details and to get started.**