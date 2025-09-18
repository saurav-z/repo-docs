<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: A Multimodal Agent with Human-Like Long-Term Memory</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

**M3-Agent empowers multimodal agents with long-term memory, enabling them to understand and reason about complex environments like humans.**

[View the original repository on GitHub](https://github.com/ByteDance-Seed/m3-agent)

## Key Features

*   **Multimodal Understanding:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Builds and updates both episodic and semantic memory for enhanced understanding.
*   **Entity-Centric Memory:** Organizes memory in a multimodal graph format for a deeper understanding of the environment.
*   **Iterative Reasoning:** Performs multi-turn reasoning and retrieves relevant information from memory to accomplish tasks.
*   **M3-Bench Dataset:** Includes M3-Bench-robot and M3-Bench-web, a novel benchmark for evaluating memory effectiveness and reasoning.
*   **Superior Performance:** Outperforms leading baselines, including prompting agents using Gemini-1.5-pro and GPT-4o, on the M3-Bench benchmark.

## Overview

M3-Agent is a cutting-edge multimodal agent designed to mimic human-like cognitive abilities. It excels at processing visual and auditory information, constructing both episodic and semantic memories. The agent structures its knowledge in a multimodal graph, fostering a robust and consistent understanding of its environment. This design allows M3-Agent to perform complex tasks by iteratively reasoning and retrieving information from its long-term memory.

## M3-Bench: Evaluating Long-Term Memory

M3-Agent's capabilities are rigorously evaluated using the M3-Bench dataset. This dataset features:

*   **M3-Bench-robot:** 100 real-world videos from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos spanning diverse scenarios.
*   **Question-Answer Pairs:** Designed to test human understanding, general knowledge extraction, and cross-modal reasoning.

![M3-Bench Example](figs/m3-bench-example.png)

**M3-Bench provides a comprehensive platform for assessing the effectiveness of multimodal agents' long-term memory and reasoning capabilities.**

**Examples from M3-Bench:** [link1](https://www.youtube.com/watch?v=7W0gRqCRMZQ), [link2](https://www.youtube.com/watch?v=Efk3K4epEzg), [link3](https://www.youtube.com/watch?v=6Unxpxy-Ct4)

![M3-Bench Statistics](figs/m3-bench-statistic.png)

### Access the Data

*   **M3-Bench-robot:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)
*   **M3-Bench-web:** Find video URLs in `data/annotations/web.json`

### Intermediate Outputs & Memory Graphs

*   **Download Directly:** Download pre-processed intermediate outputs and memory graphs from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) and [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs).
*   **Generate Manually:** Instructions for generating these outputs from video are provided in the "Run Locally" section.

## M3-Agent Architecture

![M3-Agent Architecture](figs/m3-agent.png)

M3-Agent's architecture is built around two parallel processes:

*   **Memorization:** Processes video and audio streams to create episodic and semantic memory.
*   **Control:** Executes instructions by iteratively reasoning and retrieving information from long-term memory, structured as a multimodal graph.

## Experimental Results

![Experimental Results](figs/exp_result.png)

**M3-Agent achieves significant performance gains on M3-Bench-robot, M3-Bench-web, and VideoMME-long, demonstrating its advanced capabilities.**

## Run Locally

**Before starting, ensure you add the API configuration in `configs/api_config.json`.**

### Memorization

1.  **Set up your environment:**

    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```

2.  **Cut Videos (if needed):**

    This step prepares videos for processing. The script provided cuts videos into 30-second segments.

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

    Create a `data/data.jsonl` file, with each line containing data for one video, as follows:

    ```json
    {"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
    ```

4.  **Generate Intermediate Outputs (optional):**

    **This step uses face detection and speaker diarization tools.** If you've downloaded `intermediate_outputs` from Hugging Face, you can skip this.

    *   Download the audio embedding model `pretrained_eres2netv2.ckpt` from [modelscope](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt) and save it to `models\`.
    *   Download [speakerlab](https://github.com/modelscope/3D-Speaker/tree/main/speakerlab)

    ```bash
    python m3_agent/memorization_intermediate_outputs.py \
       --data_file data/data.jsonl
    ```

5.  **Generate Memory Graphs:**

    **This step uses the M3-Agent-Memorization model.**  Download M3-Agent-Memorization from [Hugging Face](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)

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

1.  **Set up the environment:**

    ```bash
    bash setup.sh
    pip install transformers==4.51.0
    pip install vllm==0.8.4
    pip install numpy==1.26.4
    ```

2.  **Question Answering and Evaluation:**

    **This uses the M3-Agent-Control model.**  Download M3-Agent-Control from [Hugging Face](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)

    ```bash
    python m3_agent/control.py \
       --data_file data/annotations/robot.json
    ```

### Using Other Models

You can modify the code to use different models by changing the API calls and prompts:

1.  **Memorization Prompts:**
    *   Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
    *   Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`

2.  **Control Prompts:**
    *   GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training

*   **Memorization:** Refer to [sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
*   **Control:** Refer to [M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

## Citation

If you use M3-Agent in your research, please cite our work:

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