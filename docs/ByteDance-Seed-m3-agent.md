<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

# M3-Agent: A Multimodal Agent with Long-Term Memory

**Unlock human-like understanding with M3-Agent, a cutting-edge multimodal agent that sees, hears, remembers, and reasons like never before!**

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

**[Original Repository](https://github.com/ByteDance-Seed/m3-agent)**

## Key Features

*   **Multimodal Processing:** M3-Agent seamlessly integrates visual and auditory inputs for a comprehensive understanding of its environment.
*   **Long-Term Memory:**  M3-Agent builds and updates a persistent, entity-centric memory, enabling it to retain and recall information over time.
*   **Semantic Understanding:**  The agent develops semantic memory, allowing it to accumulate world knowledge and reason effectively.
*   **Multimodal Reasoning:**  M3-Agent utilizes its memory to perform multi-turn reasoning and accurately answer questions.
*   **M3-Bench Benchmark:**  Evaluated on M3-Bench, a novel long-video question-answering benchmark, demonstrating superior performance.

## What is M3-Agent?

M3-Agent is a groundbreaking multimodal agent designed to mimic human-like cognitive abilities. By processing real-time visual and auditory data, the agent constructs and maintains a long-term memory. This memory isn't just episodic; it evolves into semantic understanding, enabling sophisticated reasoning and knowledge application.

## Explore the Demo!

[Watch the M3-Agent in action!](https://www.youtube.com/watch?v=XUx31cBanfo)

## M3-Bench: Evaluating Memory and Reasoning

M3-Bench is a novel benchmark designed to assess the effectiveness of long-term memory and reasoning capabilities in multimodal agents. The benchmark includes:

*   **M3-Bench-robot:**  100 real-world videos recorded from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos spanning diverse scenarios.
*   **Question-Answer Pairs:** Designed to test key agent capabilities like human understanding, knowledge extraction, and cross-modal reasoning.

[Example Videos from M3-Bench](https://www.youtube.com/watch?v=7W0gRqCRMZQ, https://www.youtube.com/watch?v=Efk3K4epEzg, https://www.youtube.com/watch?v=6Unxpxy-Ct4)

![M3-Bench Statistic](figs/m3-bench-statistic.png)

### Get Started with M3-Bench

*   **Download Videos:**
    *   M3-Bench-robot: [Hugging Face Dataset](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)
    *   M3-Bench-web: Access video URLs in `data/annotations/web.json`

## How to Run Locally

### 1. Set up Environment

```bash
bash setup.sh
pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
pip install qwen-omni-utils==0.0.4
```

### 2. Run Memorization

1.  **Video Segmentation:**
    *   Cut videos into 30-second segments using the provided script.

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

2.  **Prepare Data** Create a `data/data.jsonl` file with video information in JSONL format.

```json
{"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
```

3.  **Generate Intermediate Outputs:**
    *   If you haven't downloaded the `intermediate_outputs` from Hugging Face, run the following command.
    *   Requires an audio embedding model (e.g., `pretrained_eres2netv2.ckpt`) and the `speakerlab` package.

```bash
python m3_agent/memorization_intermediate_outputs.py \
   --data_file data/data.jsonl
```

4.  **Generate Memory Graphs:**
    *   This step uses the M3-Agent-Memorization model.

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

### 3. Run Control

1.  **Set up Environment:**

```bash
bash setup.sh
pip install transformers==4.51.0
pip install vllm==0.8.4
pip install numpy==1.26.4
```

2.  **Question Answering and Evaluation:**
    *   This uses the M3-Agent-Control model for generating answers and GPT-4o for evaluation.

```bash
python m3_agent/control.py \
   --data_file data/annotations/robot.json
```

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