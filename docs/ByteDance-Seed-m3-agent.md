<div align="left">
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: A Multimodal Agent with Human-Like Long-Term Memory</h1>

M3-Agent is a groundbreaking multimodal agent that excels at understanding and reasoning, just like humans, by leveraging long-term memory for complex tasks.  Explore the original repository [here](https://github.com/ByteDance-Seed/m3-agent).

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

## Key Features

*   **Multimodal Understanding:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Builds and updates both episodic and semantic memory.
*   **Entity-Centric Memory:** Organizes memory in a structured, multimodal graph for deeper understanding.
*   **Iterative Reasoning:** Performs multi-turn reasoning and retrieves relevant information from memory.
*   **Superior Performance:** Outperforms state-of-the-art models on the M3-Bench benchmark.

## Overview

M3-Agent is a cutting-edge multimodal agent designed to mimic human-like cognitive abilities. It uses a unique entity-centric, multimodal memory structure to understand and reason about its environment, making it ideal for complex tasks.

![illustration](figs/illustration.png)

## Demo

Witness M3-Agent in action as a personal assistant in this demo video:

[![Watch the video](figs/demo.png)](https://www.youtube.com/watch?v=XUx31cBanfo)

Also available on [Bilibili](https://www.bilibili.com/video/BV1h9YpznEx9/)

## M3-Bench Benchmark

M3-Bench is a novel benchmark designed to evaluate the effectiveness of multimodal agents in long-term memory and reasoning tasks.  It includes two subsets:

*   **M3-Bench-robot:** 100 real-world videos from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos covering diverse scenarios.

![architecture](figs/m3-bench-example.png)

[Link 1](https://www.youtube.com/watch?v=7W0gRqCRMZQ), [Link 2](https://www.youtube.com/watch?v=Efk3K4epEzg), [Link 3](https://www.youtube.com/watch?v=6Unxpxy-Ct4)

**Key Features of M3-Bench:**

*   Long video QA designed to evaluate long-term memory capabilities
*   Robot and Web video subsets for broader evaluation
*   QA designed to test understanding, extraction, and reasoning

![architecture](figs/m3-bench-statistic.png)

## Getting Started

### Videos

1.  Download M3-Bench-robot from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)
2.  Download M3-Bench-web from the `video_url` in `data/annotations/web.json`

### Intermediate Outputs & Memory Graphs

*  **[Optional]** Download processed intermediate outputs and memory graphs from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) and [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs), respectively.

Or generate them by following the steps below.

## Run Locally

>   Before running, add the API config in `configs/api_config.json`.

### Memorization

Generate memory graphs for each video (results saved in `data/memory_graphs`).

**Prerequisites** (if you haven't downloaded intermediate_outputs and memory_graphs from Hugging Face):

1.  **Set Up Environment:**

```bash
bash setup.sh
pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
pip install qwen-omni-utils==0.0.4
```

2.  **Cut Video:**

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

3.  **Prepare Data:**  Create a JSONL file (`data/data.jsonl`) with video information:

```json
{"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
```

4.  **Generate Intermediate Outputs:**

    *   **Note:** This step uses face detection and speaker diarization.
    *   If you've downloaded `intermediate_outputs` from Hugging Face, skip this step.
    *   Download the audio embedding model (e.g., `pretrained_eres2netv2.ckpt`) from [here](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt) and save to `models/`.
    *   Download [speakerlab](https://github.com/modelscope/3D-Speaker/tree/main/speakerlab).

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

5.  **Generate Memory Graphs:**

    *   **Note:** This step uses the M3-Agent-Memorization model.
    *   Download M3-Agent-Memorization from [Hugging Face](https://huggingface.co/ByteDance-Seed/M3-Bench/tree/main/videos/robot).

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

1.  **Set Up Environment:**

```bash
bash setup.sh
pip install transformers==4.51.0
pip install vllm==0.8.4
pip install numpy==1.26.4
```

2.  **Question Answering and Evaluation:**

    *   **Note:** This step uses the M3-Agent-Control model and GPT-4o for answer generation and evaluation.
    *   Download M3-Agent-Control from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/blob/main/videos/robot).

```bash
python m3_agent/control.py \
   --data_file data/annotations/robot.json
```

### Other Models

You can adapt this process for other models by modifying the model inference to use API calls and by using the appropriate prompts.

**Prompts:**

1.  Memorization:
    *   Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
    *   Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`
2.  Control:
    *   GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training

1.  Memorization:  See [sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker).
2.  Control: See [M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training).

## Citation

If you use M3-Agent, please cite our work:

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