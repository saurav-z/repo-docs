<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: A Multimodal Agent with Human-Like Memory for Advanced AI</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

**M3-Agent is a cutting-edge multimodal agent designed to see, hear, remember, and reason, mimicking human-like cognitive abilities for complex tasks.**  [Explore the original repository](https://github.com/ByteDance-Seed/m3-agent).

**Key Features:**

*   **Multimodal Perception:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Builds and updates a dynamic long-term memory, encompassing both episodic and semantic knowledge.
*   **Entity-Centric Memory:** Organizes memory in a multimodal, entity-centric format for deeper understanding.
*   **Advanced Reasoning:** Performs multi-turn, iterative reasoning and retrieves relevant information from memory to accomplish tasks.
*   **M3-Bench Benchmark:** Includes a new benchmark (M3-Bench) for evaluating memory effectiveness and reasoning capabilities.

## What is M3-Agent?

M3-Agent represents a significant advancement in multimodal AI agents.  It is designed to understand its environment through sight and sound, retain this information in a structured memory, and then use this memory to reason and solve problems. Think of it as an AI assistant that can "learn" from its experiences, much like a human.

## M3-Bench: Evaluating Memory and Reasoning

M3-Bench is a specialized long-video question-answering benchmark developed to evaluate M3-Agent's effectiveness. It challenges agents to perform reasoning based on long-term memory.

*   **M3-Bench-robot:** Uses real-world videos recorded from a robot's perspective.
*   **M3-Bench-web:**  Employs web-sourced videos covering a wide variety of scenarios.

![architecture](figs/m3-bench-example.png)
Examples from M3-Bench. M3-Bench-robot features long videos from realistic robotic work scenarios, while M3-Bench-web expands the video diversity to support broader evaluation. The question-answering tasks are designed to assess a multimodal agent’s ability to construct consistent and reliable long-term memory, as well as to reason effectively over that memory.

![architecture](figs/m3-bench-statistic.png)
Statistical overview of M3-Bench benchmark. Each question may correspond to multiple question types.

### Accessing the Data
*   **M3-Bench-robot:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)
*   **M3-Bench-web:** Access video URLs in `data/annotations/web.json`

### Intermediate Outputs
**[Optional]** Download intermediate outputs from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) or generate them.

### Memory Graphs
**[Optional]** Download memory graphs from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs) or generate them.

## M3-Agent Architecture

![architecture](figs/m3-agent.png)

M3-Agent's architecture consists of two core processes:

*   **Memorization:** Processes video and audio streams to create episodic and semantic memory.
*   **Control:** Executes instructions by reasoning and retrieving information from long-term memory. The long-term memory is structured as a multimodal graph.

## Experimental Results

![architecture](figs/exp_result.png)

M3-Agent demonstrates strong performance on M3-Bench-robot, M3-Bench-web, and VideoMME-long, outperforming baseline models.

## Running M3-Agent Locally

Follow these steps to run M3-Agent:

> Before running, add api config in `configs/api_config.json`

### 1. Setup Environment

```bash
bash setup.sh
pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
pip install qwen-omni-utils==0.0.4
```

### 2. Memorization

#### 2.1. Cut Video (If not using M3-Bench videos or have pre-cut clips)

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

#### 2.2. Prepare Data

Create a `data/data.jsonl` file:

```json
{"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
```

#### 2.3. Generate Intermediate Outputs (if not downloaded)

**This step utilizes Face Detection and Speaker Diarization.**

*   Download audio embedding model from [pretrained_eres2netv2.ckpt](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt)
*   Download speakerlab from [3D-Speaker](https://github.com/modelscope/3D-Speaker/tree/main/speakerlab)

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

#### 2.4. Generate Memory Graphs (using M3-Agent-Memorization model)

*   Download M3-Agent-Memorization from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)

```bash
python m3_agent/memorization_memory_graphs.py \
   --data_file data/data.jsonl
```

#### 2.5. Memory Graph Visualization

```bash
python visualization.py \
   --mem_path data/memory_graphs/robot/bedroom_01.pkl \
   --clip_id 1
```

### 3. Control

#### 3.1. Setup Environment

```bash
bash setup.sh
pip install transformers==4.51.0
pip install vllm==0.8.4
pip install numpy==1.26.4
```

#### 3.2. Question Answering and Evaluation

*   Download M3-Agent-Control from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/blob/main/videos/robot)

```bash
python m3_agent/control.py \
   --data_file data/annotations/robot.json
```

### 4. Using Other Models

Easily integrate other models by adjusting model inference calls and prompts.

**Prompts:**

1.  **Memorization:**
    *   Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
    *   Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`

2.  **Control:**
    *   GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training Resources

*   Memorization: [sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
*   Control: [M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

## Citation

If you use M3-Agent in your research, please cite:

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