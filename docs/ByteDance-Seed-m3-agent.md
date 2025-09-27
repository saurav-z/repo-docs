<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: A Multimodal Agent with Long-Term Memory for Advanced Reasoning</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

**M3-Agent is a groundbreaking multimodal agent that learns, remembers, and reasons like humans, enabling advanced understanding and interaction.**

## Key Features

*   **Multimodal Perception:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Builds and updates an entity-centric, multimodal memory, including episodic and semantic knowledge.
*   **Advanced Reasoning:**  Performs multi-turn, iterative reasoning and retrieves relevant information from memory for task completion.
*   **M3-Bench Dataset:** Includes M3-Bench-robot (robot-perspective videos) and M3-Bench-web (diverse web videos) to evaluate memory and reasoning capabilities.
*   **Superior Performance:** Outperforms leading models like Gemini-1.5-pro and GPT-4o on M3-Bench and VideoMME-long.

## Overview

M3-Agent emulates human cognitive processes by integrating perception, memory, and reasoning. The agent's architecture comprises two key components: memorization, which captures information from visual and audio streams, and control, which uses memory for decision-making. The agent's effectiveness is demonstrated on the M3-Bench dataset, a novel benchmark designed to evaluate the abilities of multimodal agents to perform reasoning over long-term memory.

## M3-Bench Dataset

The M3-Bench dataset provides the foundation for evaluating and improving multimodal agents. It features two distinct subsets:

*   **M3-Bench-robot:** Real-world videos recorded from a robot's perspective, focusing on realistic scenarios.
*   **M3-Bench-web:** A diverse set of web-sourced videos covering a wide range of content and tasks.

Each instance includes long-form videos paired with open-ended question-answer pairs, designed to test the agent's ability to understand context, extract knowledge, and perform cross-modal reasoning.

![architecture](figs/m3-bench-example.png)
Examples from M3-Bench. M3-Bench-robot features long videos from realistic robotic work scenarios, while M3-Bench-web expands the video diversity to support broader evaluation. The question-answering tasks are designed to assess a multimodal agent’s ability to construct consistent and reliable long-term memory, as well as to reason effectively over that memory.

![architecture](figs/m3-bench-statistic.png)
Statistical overview of M3-Bench benchmark. Each question may correspond to multiple question types.

### Dataset Resources

*   **M3-Bench-robot:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)
*   **M3-Bench-web:** Access video URLs in `data/annotations/web.json`

### Intermediate Outputs & Memory Graphs

**Optional:** Accelerate your workflow by using pre-processed intermediate outputs and memory graphs. Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) and [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs), respectively. Alternatively, generate these directly:

## M3-Agent Architecture

![architecture](figs/m3-agent.png)
The M3-Agent architecture uses two parallel processes: memorization (builds memory from video and audio) and control (executes instructions by retrieving information from long-term memory). Long-term memory is structured as a multimodal graph.

## Experimental Results

![architecture](figs/exp_result.png)
Key performance results across M3-Bench-robot, M3-Bench-web, and VideoMME-long, highlighting M3-Agent's superior performance.

## Run Locally

Detailed instructions for setting up the environment, generating memory graphs, and performing control tasks.

### Memorization

Steps to generate memory graphs.  Requires video pre-processing and use of the M3-Agent-Memorization model.

1.  **Set up environment:**
    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```

2.  **Video Segmentation (Cut Video):** Use the provided script to split videos into 30-second segments:

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

3.  **Prepare Data:** Create a `data/data.jsonl` file with video information:

```json
{"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
```

4.  **Generate Intermediate Outputs:**  This step uses face detection and speaker diarization:

    *   Download the audio embedding model and save it in `models\` (pretrained_eres2netv2.ckpt from [here](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt))
    *   Download speakerlab:

```bash
m3-agent
├── models
│   └── pretrained_eres2netv2.ckpt
└── speakerlab
```

```bash
python m3_agent/memorization_intermediate_outputs.py \
   --data_file data/data.jsonl
```

5.  **Generate Memory Graphs:** Use the M3-Agent-Memorization model.

    *   Download M3-Agent-Memorization from [Hugging Face](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
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

1.  **Set up environment:**

```bash
bash setup.sh
pip install transformers==4.51.0
pip install vllm==0.8.4
pip install numpy==1.26.4
```

2.  **Question Answering and Evaluation:**  Use the M3-Agent-Control model and GPT-4o for evaluation.

    *   Download M3-Agent-Control from [Hugging Face](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)

```bash
python m3_agent/control.py \
   --data_file data/annotations/robot.json
```

### Using Other Models

Easily adapt prompts for alternative models to generate memory or answer questions.

*   **Prompts for Memorization:**
    *   Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
    *   Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`

*   **Prompts for Control:**
    *   GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training

*   **Memorization:**  See [sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
*   **Control:** See [M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

## Citation

If you use M3-Agent, please cite our paper:

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

**[View the original repository here](https://github.com/ByteDance-Seed/m3-agent)**