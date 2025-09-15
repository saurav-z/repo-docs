<div align="left">
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: A Multimodal Agent with Human-Like Long-Term Memory</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

**M3-Agent is a groundbreaking multimodal agent designed to understand, remember, and reason like humans, paving the way for more intelligent and adaptable AI systems.** Explore the project on [GitHub](https://github.com/ByteDance-Seed/m3-agent).

## Key Features

*   **Multimodal Understanding:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Builds and updates a rich long-term memory, including episodic and semantic components.
*   **Entity-Centric Memory:** Organizes information in an entity-centric, multimodal format for deeper understanding.
*   **Iterative Reasoning:** Performs multi-turn reasoning, retrieving relevant information from memory to accomplish tasks.
*   **High Performance:** Outperforms state-of-the-art baselines on the M3-Bench dataset.
*   **Open Source:** Provides access to the M3-Bench Dataset, models, and code for research.

## Introduction to M3-Agent

M3-Agent represents a significant advancement in multimodal AI, mimicking human cognitive abilities by integrating long-term memory with real-time perception.  It excels at understanding complex scenes, remembering details over time, and reasoning about the world, which is critical for advanced AI applications.

## M3-Bench: A Benchmark for Long-Term Memory Reasoning

To evaluate M3-Agent's capabilities, the project introduces **M3-Bench**, a novel long-video question-answering benchmark.

*   **M3-Bench-robot:** Contains 100 videos from a robot's perspective.
*   **M3-Bench-web:** Includes 920 web-sourced videos across diverse scenarios.
*   Question-answer pairs test key capabilities like understanding humans, extracting general knowledge, and cross-modal reasoning.

![M3-Bench Example](figs/m3-bench-example.png)
*Examples from M3-Bench. M3-Bench-robot features long videos from realistic robotic work scenarios, while M3-Bench-web expands the video diversity to support broader evaluation. The question-answering tasks are designed to assess a multimodal agent’s ability to construct consistent and reliable long-term memory, as well as to reason effectively over that memory.*

![M3-Bench Statistic Overview](figs/m3-bench-statistic.png)
*Statistical overview of M3-Bench benchmark. Each question may correspond to multiple question types.*

### Accessing M3-Bench Data

*   **M3-Bench-robot Videos:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot).
*   **M3-Bench-web Videos:**  Download from the video_url links in `data/annotations/web.json`.

### Intermediate Outputs and Memory Graphs

**Optionally, you can download pre-processed data from Hugging Face or generate it yourself.**

*   **Intermediate Outputs:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) or generate them using the steps in the "Run Locally" section.
*   **Memory Graphs:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs) or generate them using the steps in the "Run Locally" section.

## M3-Agent Architecture

![M3-Agent Architecture](figs/m3-agent.png)

The M3-Agent architecture consists of two parallel processes:

*   **Memorization:** Processes video and audio streams to generate episodic and semantic memory.
*   **Control:** Executes instructions by iteratively reasoning and retrieving information from long-term memory.
*   Long-term memory is structured as a multimodal graph.

## Experimental Results

![Experimental Results](figs/exp_result.png)
*Results on M3-Bench-robot, M3-Bench-web, and VideoMME-long.*

M3-Agent demonstrates superior performance on the M3-Bench and VideoMME-long benchmarks compared to the state-of-the-art baselines.

## Run M3-Agent Locally

This section details the steps to set up and run the M3-Agent.

### Prerequisites

*   Add API configuration in `configs/api_config.json`.

### 1. Memorization

Generate memory graphs for each video. The results are saved in `data/memory_graphs`.

*   These steps are required only if you haven't downloaded *intermediate_outputs* and *memory_graphs* from huggingface or want to process other videos not from M3-Bench.

1.  **Set Up Environment:**

    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```

2.  **Cut Video:**

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

3.  **Prepare Data:**

    Prepare a jsonl file with one video per line saved in `data/data.jsonl`

    ```json
    {"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
    ```

4.  **Generate Intermediate Outputs:**

    **This step uses Face Detection and Speaker Diarization tools to generate intermediate outputs.**

    *   If you want to use M3-Bench and have downloaded intermediate_outputs from huggingface, you can skip this step.

    *   Download audio embedding model and save into `models\` from [pretrained_eres2netv2.ckpt](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt)

    *   Download [speakerlab](https://github.com/modelscope/3D-Speaker/tree/main/speakerlab)

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

    **This step uses the M3-Agent-Memorization model to generate memory graphs.**

    *   Download M3-Agent-Memorization from [huggingface](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)

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

### 2. Control

1.  **Set Up Environment:**

    ```bash
    bash setup.sh
    pip install transformers==4.51.0
    pip install vllm==0.8.4
    pip install numpy==1.26.4
    ```

2.  **Question Answering and Evaluation:**

    **This step uses the M3-Agent-Control model to generate the answer and GPT-4o to evaluate the answer.**

    *   Download M3-Agent-Control from [huggingface](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)

    ```bash
    python m3_agent/control.py \
       --data_file data/annotations/robot.json
    ```

### 3. Other Models

You can prompt other models to generate memory or answer questions. The prompt needs to change into API calling and use the corresponding prompts.

**Prompts:**

1.  **Memorization**

    *   Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
    *   Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`

2.  **Control**

    *   GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training

Training resources are available:

1.  **Memorization:** [https://github.com/hyc2026/sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
2.  **Control:** [https://github.com/hyc2026/M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

## Citation

If you find this work helpful, please cite the project:

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