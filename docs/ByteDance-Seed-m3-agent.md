<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

# M3-Agent: A Multimodal Agent with Long-Term Memory for Advanced Reasoning

**M3-Agent** is a groundbreaking multimodal agent that learns to see, hear, and remember, enabling advanced reasoning capabilities, and you can explore the project on the original repository: [ByteDance-Seed/m3-agent](https://github.com/ByteDance-Seed/m3-agent).

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

## Key Features

*   **Multimodal Perception:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Builds and updates both episodic and semantic memory.
*   **Entity-Centric Memory:** Organizes memory for deeper understanding.
*   **Autonomous Reasoning:** Performs multi-turn reasoning and information retrieval.
*   **Superior Performance:** Outperforms strong baselines on the M3-Bench benchmark.

## M3-Bench: Evaluating Long-Term Memory in Multimodal Agents

M3-Bench is a novel benchmark designed to evaluate the effectiveness of memory and reasoning in multimodal agents using long videos.

### M3-Bench Subsets
1.  **M3-Bench-robot:** 100 real-world videos from a robot's perspective.
2.  **M3-Bench-web:** 920 web-sourced videos covering diverse scenarios.

![architecture](figs/m3-bench-example.png)
[link1](https://www.youtube.com/watch?v=7W0gRqCRMZQ), [link2](https://www.youtube.com/watch?v=Efk3K4epEzg), [link3](https://www.youtube.com/watch?v=6Unxpxy-Ct4)\
Examples from M3-Bench. M3-Bench-robot features long videos from realistic robotic work scenarios, while M3-Bench-web expands the video diversity to support broader evaluation. The question-answering tasks are designed to assess a multimodal agent’s ability to construct consistent and reliable long-term memory, as well as to reason effectively over that memory.

![architecture](figs/m3-bench-statistic.png)

Statistical overview of M3-Bench benchmark. Each question may correspond to multiple question types.

### Accessing M3-Bench Data

1.  **M3-Bench-robot Videos:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot).
2.  **M3-Bench-web Videos:** Access via video_url in `data/annotations/web.json`.

### Intermediate Outputs & Memory Graphs

**Optional:** You can either download preprocessed data or generate it yourself.

*   **Intermediate Outputs:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs).
*   **Memory Graphs:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs).

## M3-Agent Architecture

![architecture](figs/m3-agent.png)

M3-Agent operates with two parallel processes: memorization and control. Memorization generates episodic and semantic memory from video and audio streams. Control executes instructions through iterative reasoning and memory retrieval, using a multimodal graph structure for long-term memory.

## Experimental Results

![architecture](figs/exp_result.png)

M3-Agent achieves superior performance on M3-Bench-robot, M3-Bench-web, and VideoMME-long.

## Run Locally

### Memorization

Follow the instructions below to generate memory graphs.
1.  Set up environment
```bash
bash setup.sh
pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
pip install qwen-omni-utils==0.0.4
```

2.  Cut Video
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

3.  Prepare data
```json
{"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
```

4.  Generate Intermediate Outputs
```bash
python m3_agent/memorization_intermediate_outputs.py \
   --data_file data/data.jsonl
```
5.  Generate Memory Graphs
```bash
python m3_agent/memorization_memory_graphs.py \
   --data_file data/data.jsonl
```
6.  Memory Graph Visualization
```bash
python visualization.py \
   --mem_path data/memory_graphs/robot/bedroom_01.pkl \
   --clip_id 1
```

### Control
1. Set up environment
```bash
bash setup.sh
pip install transformers==4.51.0
pip install vllm==0.8.4
pip install numpy==1.26.4
```

2. Question Answering and Evaluation
```bash
python m3_agent/control.py \
   --data_file data/annotations/robot.json
```

### Other Models
```
1. Memorization
   - Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
   - Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`

2. Control
   - GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`
```

## Training

1.  Memorization: [https://github.com/hyc2026/sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
2.  Control: [https://github.com/hyc2026/M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

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