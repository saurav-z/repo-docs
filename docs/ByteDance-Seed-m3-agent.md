<div align="left">
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: A Multimodal Agent with Long-Term Memory for Advanced Reasoning</h1>

**M3-Agent is a groundbreaking multimodal agent capable of processing visual and auditory information to build, update, and reason over long-term memory, offering a significant leap in human-like AI.**  Explore the capabilities of this innovative agent and its performance on the M3-Bench benchmark.

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

**Key Features:**

*   **Multimodal Processing:** Integrates visual and auditory inputs for a comprehensive understanding of the environment.
*   **Long-Term Memory:** Develops both episodic and semantic memory for continuous learning and knowledge accumulation.
*   **Entity-Centric Memory:** Organizes memory in a structured, multimodal graph format for deeper and more consistent understanding.
*   **Autonomous Reasoning:**  Performs iterative reasoning and retrieves relevant information from memory to accomplish tasks.
*   **M3-Bench Benchmark:** Introduces a novel benchmark for evaluating memory effectiveness and reasoning in multimodal agents.

**[See the original repository](https://github.com/ByteDance-Seed/m3-agent)**

## Core Components

### M3-Bench: Evaluating Long-Term Memory in Multimodal Agents

M3-Bench is a dataset designed to evaluate the ability of multimodal agents to reason over long-term memory through long-video question answering.

*   **M3-Bench-robot:**  Contains 100 real-world videos recorded from a robot's perspective.
*   **M3-Bench-web:**  Includes 920 web-sourced videos covering diverse scenarios.

**Example:**

![architecture](figs/m3-bench-example.png)

**Dataset Statistics:**

![architecture](figs/m3-bench-statistic.png)

#### Accessing the M3-Bench Dataset

*   **M3-Bench-robot Videos:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)
*   **M3-Bench-web Videos:** Access via video URLs in `data/annotations/web.json`
*   **Intermediate Outputs & Memory Graphs:** [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench) (Optional - for direct use, or to generate your own)

### M3-Agent Architecture

M3-Agent utilizes a dual-process architecture: memorization and control.

![architecture](figs/m3-agent.png)

*   **Memorization:** Processes video and audio streams to generate episodic and semantic memory.
*   **Control:** Executes instructions by iteratively thinking and retrieving information from long-term memory.

### Experimental Results

The M3-Agent demonstrates superior performance on M3-Bench and VideoMME-long.

![architecture](figs/exp_result.png)

## Run Locally

Follow these instructions to set up and run the M3-Agent.

**Prerequisites:**

*   API configuration in `configs/api_config.json`

### 1. Memorization

To generate memory graphs for each video:

**Steps (Required if not using pre-downloaded intermediate and memory graph outputs, or to process other videos):**

1.  **Set up Environment:**
    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```

2.  **Video Segmentation (Optional):**
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
    Create a JSONL file `data/data.jsonl`:
    ```json
    {"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
    ```

4.  **Generate Intermediate Outputs:**
    **This step uses Face Detection and Speaker Diarization tools to generate intermediate outputs.**

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

1.  **Set up Environment:**

    ```bash
    bash setup.sh
    pip install transformers==4.51.0
    pip install vllm==0.8.4
    pip install numpy==1.26.4
    ```

2.  **Question Answering and Evaluation:**
    **This step uses the M3-Agent-Control model to generate answers and uses GPT-4o to evaluate the answers.**
    *   Download M3-Agent-Control from [huggingface](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)

    ```bash
    python m3_agent/control.py \
       --data_file data/annotations/robot.json
    ```

### Other Models

You can modify prompts to use other models for memory generation or question answering.

**Prompts:**

1.  **Memorization:**
    *   Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
    *   Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`
2.  **Control:**
    *   GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training

*   Memorization: [sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
*   Control: [M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

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