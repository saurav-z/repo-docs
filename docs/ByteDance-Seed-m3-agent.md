<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: A Multimodal Agent with Human-Like Memory and Reasoning</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

**M3-Agent revolutionizes multimodal understanding by equipping agents with long-term memory, enabling them to learn and reason like humans.**  For the original repo, check out the [M3-Agent GitHub](https://github.com/ByteDance-Seed/m3-agent).

## Key Features

*   **Long-Term Memory:** M3-Agent builds and updates its long-term memory by processing real-time visual and auditory inputs.
*   **Entity-Centric Memory:**  Memory is organized in a multimodal graph, enabling deeper understanding of the environment.
*   **Multimodal Reasoning:**  M3-Agent performs multi-turn, iterative reasoning and retrieves relevant information from memory to complete tasks.
*   **M3-Bench:**  Includes M3-Bench, a novel benchmark with long-video question-answering tasks to evaluate memory effectiveness and reasoning capabilities in multimodal agents.

## M3-Bench Dataset

M3-Bench is designed to evaluate the ability of multimodal agents to reason over long-term memory. It includes:

*   **M3-Bench-robot:** 100 real-world videos from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos covering diverse scenarios.

[Video Examples from M3-Bench](https://www.youtube.com/watch?v=7W0gRqCRMZQ), [Video Examples from M3-Bench](https://www.youtube.com/watch?v=Efk3K4epEzg), [Video Examples from M3-Bench](https://www.youtube.com/watch?v=6Unxpxy-Ct4)

![architecture](figs/m3-bench-statistic.png)

## M3-Agent Architecture

![architecture](figs/m3-agent.png)

The M3-Agent system comprises two parallel processes:

*   **Memorization:** Processes video and audio streams to generate episodic and semantic memory.
*   **Control:** Executes instructions by iteratively thinking and retrieving information from long-term memory.

## Experimental Results

![architecture](figs/exp_result.png)

M3-Agent demonstrates superior performance on the M3-Bench benchmark, including M3-Bench-robot, M3-Bench-web, and VideoMME-long.

## Run Locally

Before running, add API config in `configs/api_config.json`.

### Memorization

1.  **Setup Environment:**
    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```

2.  **Cut Video**
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
3.  **Prepare Data:** Create a `data.jsonl` file with video information.

    ```json
    {"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
    ```
4.  **Generate Intermediate Outputs:** (Optional, if not downloaded from Hugging Face)
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

5.  **Generate Memory Graphs:** (Requires the M3-Agent-Memorization model from Hugging Face)

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

1.  **Set up environment**

    ```bash
    bash setup.sh
    pip install transformers==4.51.0
    pip install vllm==0.8.4
    pip install numpy==1.26.4
    ```
2.  **Question Answering and Evaluation:** (Requires the M3-Agent-Control model from Hugging Face)

    ```bash
    python m3_agent/control.py \
       --data_file data/annotations/robot.json
    ```

### Other Models

You can integrate other models by modifying the API calls with corresponding prompts for memorization and control tasks.

*   **Memorization Prompts:** Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`; Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`
*   **Control Prompts:** GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training

1.  Memorization: [sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
2.  Control: [M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

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
```