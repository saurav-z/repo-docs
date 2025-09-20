<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

# M3-Agent: Your Multimodal AI Assistant with a Long-Term Memory

**M3-Agent is a cutting-edge multimodal agent designed to see, listen, remember, and reason, providing a more human-like understanding of the world.**  For the latest updates and more information, please visit the original repository: [ByteDance-Seed/m3-agent](https://github.com/ByteDance-Seed/m3-agent).

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

## Key Features

*   **Long-Term Memory:** M3-Agent builds and updates its memory over time, similar to how humans remember experiences.
*   **Multimodal Understanding:** Processes visual and auditory inputs to understand its environment comprehensively.
*   **Entity-Centric Memory:** Organizes information around entities, facilitating deeper and more consistent understanding.
*   **Iterative Reasoning:** Performs multi-turn reasoning, retrieving relevant information from memory to accomplish tasks.
*   **Enhanced Performance:** Outperforms leading models on the M3-Bench benchmark.

## M3-Bench: Evaluating Multimodal Agent Capabilities

M3-Bench is a new, comprehensive benchmark designed to evaluate the effectiveness of multimodal agents with long-term memory, offering:

*   **Long-Video Question Answering:** Tests agents' reasoning abilities over extended video content.
*   **Two Subsets:**
    *   **M3-Bench-robot:** Features 100 real-world videos from a robot's perspective.
    *   **M3-Bench-web:** Includes 920 web-sourced videos across diverse scenarios.
*   **Evaluation of Key Capabilities:** Assesses human understanding, knowledge extraction, and cross-modal reasoning.

![architecture](figs/m3-bench-example.png)
_Examples from M3-Bench. M3-Bench-robot features long videos from realistic robotic work scenarios, while M3-Bench-web expands the video diversity to support broader evaluation. The question-answering tasks are designed to assess a multimodal agent’s ability to construct consistent and reliable long-term memory, as well as to reason effectively over that memory._

![architecture](figs/m3-bench-statistic.png)
_Statistical overview of M3-Bench benchmark. Each question may correspond to multiple question types._

### Accessing M3-Bench Data

*   **M3-Bench-robot Videos:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)
*   **M3-Bench-web Videos:** Access video URLs in `data/annotations/web.json`
*   **Intermediate Outputs (Optional):** Download pre-processed data from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) or generate them.
*   **Memory Graphs (Optional):** Download pre-processed memory graphs from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs) or generate them.

### Example Videos:

[link1](https://www.youtube.com/watch?v=7W0gRqCRMZQ), [link2](https://www.youtube.com/watch?v=Efk3K4epEzg), [link3](https://www.youtube.com/watch?v=6Unxpxy-Ct4)

## M3-Agent Architecture

![architecture](figs/m3-agent.png)

*   **Memorization:** Processes video and audio to create episodic and semantic memory.
*   **Control:** Executes instructions using iterative reasoning and memory retrieval.
*   **Multimodal Graph Structure:** Organizes long-term memory for efficient access and understanding.

## Experimental Results

![architecture](figs/exp_result.png)

*   Demonstrates superior performance on M3-Bench-robot, M3-Bench-web, and VideoMME-long benchmarks.

## Run M3-Agent Locally

Follow these steps to set up and run M3-Agent:

1.  **Configure API:** Add your API configuration to `configs/api_config.json`.
2.  **Memorization:**
    *   (If not using pre-processed data) Generate memory graphs for each video.
    *   **Set up the environment:**
        ```bash
        bash setup.sh
        pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
        pip install qwen-omni-utils==0.0.4
        ```

    *   **Cut Videos:**
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

    *   **Prepare Data:** Create a `data.jsonl` file with video information.
        ```json
        {"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
        ```

    *   **Generate Intermediate Outputs:**
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
    *   **Generate Memory Graphs:**
        *   Download M3-Agent-Memorization from [huggingface](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
        ```bash
        python m3_agent/memorization_memory_graphs.py \
           --data_file data/data.jsonl
        ```
    *   **Memory Graph Visualization:**
        ```bash
        python visualization.py \
           --mem_path data/memory_graphs/robot/bedroom_01.pkl \
           --clip_id 1
        ```

3.  **Control:**
    *   **Set up the environment:**
        ```bash
        bash setup.sh
        pip install transformers==4.51.0
        pip install vllm==0.8.4
        pip install numpy==1.26.4
        ```
    *   **Question Answering and Evaluation:**
        *   Download M3-Agent-Control from [huggingface](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
        ```bash
        python m3_agent/control.py \
           --data_file data/annotations/robot.json
        ```

4.  **Alternative Models:** Easily integrate other models for memory generation or question answering by adjusting API calls and prompts.

## Training Resources

*   **Memorization:** [sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
*   **Control:** [M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

## Citation

If you use M3-Agent in your research, please cite the following:

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