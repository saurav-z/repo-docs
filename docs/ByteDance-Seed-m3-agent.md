<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

# M3-Agent: A Multimodal Agent with Human-Like Long-Term Memory

**M3-Agent is a cutting-edge multimodal agent capable of processing visual and auditory inputs to build and leverage a long-term memory, excelling at reasoning and question-answering tasks.**  Explore the capabilities of this innovative agent on its [original GitHub repository](https://github.com/ByteDance-Seed/m3-agent).

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

## Key Features

*   **Multimodal Perception:** Processes and integrates real-time visual and auditory data.
*   **Long-Term Memory:** Builds and updates both episodic and semantic memory, mirroring human cognitive processes.
*   **Entity-Centric Memory:** Organizes information in a structured, multimodal graph format for enhanced understanding.
*   **Iterative Reasoning:** Performs multi-turn reasoning and retrieves relevant information from memory to complete tasks.
*   **Superior Performance:** Outperforms leading baselines like Gemini-1.5-pro and GPT-4o on the M3-Bench benchmark.
*   **Open-Source Resources**: Access models and datasets on Hugging Face.

## M3-Bench: A Benchmark for Long-Term Memory in Multimodal Agents

M3-Bench is a novel benchmark designed to evaluate the effectiveness of multimodal agents in reasoning over long-term memory. It comprises long-form videos paired with question-answer pairs, designed to test critical agent abilities.

*   **M3-Bench-robot:** 100 real-world videos recorded from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos across diverse scenarios.

![M3-Bench Example](figs/m3-bench-example.png)
Examples from M3-Bench. M3-Bench-robot features long videos from realistic robotic work scenarios, while M3-Bench-web expands the video diversity to support broader evaluation. The question-answering tasks are designed to assess a multimodal agent’s ability to construct consistent and reliable long-term memory, as well as to reason effectively over that memory.

![M3-Bench Statistics](figs/m3-bench-statistic.png)
Statistical overview of M3-Bench benchmark. Each question may correspond to multiple question types.

### Data Resources

*   **M3-Bench-robot Videos:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot).
*   **M3-Bench-web Videos:** Find video URLs in `data/annotations/web.json`.
*   **Intermediate Outputs:** (Optional) Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) or generate them.
*   **Memory Graphs:** (Optional) Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs) or generate them.

## M3-Agent Architecture

M3-Agent's architecture is designed with two parallel processes: memorization and control.  During memorization, the agent processes video and audio streams to generate episodic and semantic memory.  During control, it uses these memories to answer questions and execute commands through iterative reasoning.

![M3-Agent Architecture](figs/m3-agent.png)

## Experimental Results

M3-Agent demonstrates state-of-the-art performance, as highlighted in the experimental results below:

![Experimental Results](figs/exp_result.png)

## Run Locally

Get started by setting up your environment and running the M3-Agent on your own machine.

### Prerequisites

1.  **API Configuration:** Add your API keys in `configs/api_config.json`.

### Steps

1.  **Set up environment**
    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```
2.  **(Optional) Video Segmentation:**  Split videos into 30-second segments.
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
3.  **Prepare Data:** Create a JSONL file `data/data.jsonl` with the following structure:
    ```json
    {"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
    ```
4.  **(Optional) Generate Intermediate Outputs:** Face detection and speaker diarization.  Download the necessary models.
    ```bash
    python m3_agent/memorization_intermediate_outputs.py \
       --data_file data/data.jsonl
    ```
5.  **Generate Memory Graphs:**  Using the M3-Agent-Memorization model. Download M3-Agent-Memorization from [huggingface](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
    ```bash
    python m3_agent/memorization_memory_graphs.py \
       --data_file data/data.jsonl
    ```
6.  **(Optional) Memory Graph Visualization:** Visualize the memory graphs.
    ```bash
    python visualization.py \
       --mem_path data/memory_graphs/robot/bedroom_01.pkl \
       --clip_id 1
    ```

### Control

1.  **Set up environment**
    ```bash
    bash setup.sh
    pip install transformers==4.51.0
    pip install vllm==0.8.4
    pip install numpy==1.26.4
    ```

2.  **Question Answering and Evaluation:**  Using the M3-Agent-Control model.  Download M3-Agent-Control from [huggingface](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
    ```bash
    python m3_agent/control.py \
       --data_file data/annotations/robot.json
    ```

### Other Models

Adapt prompts for other models to generate memory or answer questions.

### Prompts

1.  **Memorization**
    *   Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
    *   Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`

2.  **Control**
    *   GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training

*   **Memorization:** Refer to [sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker).
*   **Control:**  Refer to [M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training).

## Citation

If you use M3-Agent in your research, please cite the following paper:

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