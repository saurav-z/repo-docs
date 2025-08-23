<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: A Multimodal Agent with Human-Like Long-Term Memory</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

**M3-Agent is a groundbreaking multimodal agent that emulates human-like cognitive abilities by leveraging long-term memory for enhanced understanding and reasoning.** [Learn more about M3-Agent on GitHub](https://github.com/ByteDance-Seed/m3-agent).

**Key Features:**

*   **Multimodal Perception:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Builds and updates episodic and semantic memory for robust knowledge retention.
*   **Entity-Centric Organization:** Structures memory in a multimodal graph format for deeper understanding.
*   **Advanced Reasoning:** Performs multi-turn, iterative reasoning to accomplish complex tasks.
*   **M3-Bench Benchmark:** Introduces a novel benchmark for evaluating memory effectiveness in multimodal agents.
*   **Superior Performance:** Achieves state-of-the-art results on the M3-Bench benchmark.

## Overview

M3-Agent represents a significant advancement in multimodal agent technology. It's designed to understand and interact with the world in a manner that mirrors human cognitive processes. The agent's architecture includes two primary components: Memorization and Control. Memorization processes real-time video and audio streams to generate episodic and semantic memory. Control then utilizes this memory to execute instructions through iterative reasoning.

![architecture](figs/m3-agent.png)

## M3-Bench: Evaluating Long-Term Memory

To evaluate the effectiveness of M3-Agent, we developed M3-Bench, a long-video question-answering dataset. This benchmark assesses an agent's ability to reason over long-term memory, crucial for real-world applications.

*   **M3-Bench-robot:** Contains 100 real-world videos recorded from a robot's perspective.
*   **M3-Bench-web:** Includes 920 web-sourced videos covering a wide array of content and scenarios.

![architecture](figs/m3-bench-example.png)
Examples from M3-Bench. M3-Bench-robot features long videos from realistic robotic work scenarios, while M3-Bench-web expands the video diversity to support broader evaluation. The question-answering tasks are designed to assess a multimodal agent’s ability to construct consistent and reliable long-term memory, as well as to reason effectively over that memory.

![architecture](figs/m3-bench-statistic.png)
Statistical overview of M3-Bench benchmark. Each question may correspond to multiple question types.

### Download M3-Bench
1. Download M3-Bench-robot from [huggingface](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)
2. Download M3-Bench-web from video_url in `data/annotations/web.json`

### Intermediate Outputs
**[optional]** You can either download the intermediate outputs we have processed from [huggingface](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) or generate them directly from the video by the following steps.

### Memory Graphs
**[optional]** You can either download and extract the memory graphs we have processed from [huggingface](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs) or generate them directly from the video by the following steps.

## Demo
A video demo showcasing M3-Agent's capabilities is available:

[![Watch the video](figs/demo.png)](https://www.youtube.com/watch?v=XUx31cBanfo)

You can also watch the demo on [Bilibili](https://www.bilibili.com/video/BV1h9YpznEx9/).

## Experimental Results

M3-Agent demonstrates impressive performance on the M3-Bench benchmark and VideoMME-long.

![architecture](figs/exp_result.png)
Results on M3-Bench-robot, M3-Bench-web, and VideoMME-long.

## Run Locally

Follow the steps below to run M3-Agent locally.

### Prerequisites
*   Add API config in `configs/api_config.json`

### Memorization
Generate memory graphs for each video. The results are saved in `data/memory_graphs`.
* The following steps are required only if you haven't downloaded *intermediate_outputs* and *memory_graphs* from huggingface or want to process other videos not from M3-Bench.
1. Set up environment
    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```
2. Cut Video
    Cut the video into 30-second segments.
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
3. Prepare data
    Prepare a jsonl file with one video per line saved in `data/data.jsonl`
    ```json
    {"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
    ```
4. Generate Intermediate Outputs
    **This step uses Face Detection and Speaker Diarization tools to generate intermediate outputs.**
    - If you want to use M3-Bench and have downloaded intermediate_outputs from huggingface, you can skip this step.
    - Download audio embedding model and save into `models\` from [pretrained_eres2netv2.ckpt](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt)
    - Download [speakerlab](https://github.com/modelscope/3D-Speaker/tree/main/speakerlab)
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
5. Generate Memory Graphs
    **This step uses the M3-Agent-Memorization model to generate memory graphs.**
    - Download M3-Agent-Memorization from [huggingface](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)
    ```bash
    python m3_agent/memorization_memory_graphs.py \
       --data_file data/data.jsonl
    ```
6. Memory Graph Visualization
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
    **This step uses the M3-Agent-Control model to generate answer and the GPT-4o to evaluate the answer.**
    - Download M3-Agent-Control from [huggingface](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/blob/main/videos/robot)
    ```bash
    python m3_agent/control.py \
       --data_file data/annotations/robot.json
    ```

### Other Models
If you want to prompt other models to generate memory or answer question, only need to change the model inference into api calling and use the corresponding prompt.

Prompts:
1. Memorization
    - Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
    - Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`
2. Control
    - GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training

*   Memorization: [https://github.com/hyc2026/sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
*   Control: [https://github.com/hyc2026/M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

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