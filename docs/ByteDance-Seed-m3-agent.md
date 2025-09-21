<div align="left">
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

# M3-Agent: Building Human-Like Memory in Multimodal Agents

**M3-Agent** is a groundbreaking multimodal agent capable of **seeing, listening, remembering, and reasoning**, bringing us closer to AI that mimics human cognitive abilities. Explore the project on [GitHub](https://github.com/ByteDance-Seed/m3-agent).

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

## Key Features

*   **Multimodal Input Processing:** Integrates visual and auditory inputs for comprehensive understanding.
*   **Long-Term Memory:** Constructs and updates a rich, entity-centric memory, capturing both episodic and semantic information.
*   **Contextual Reasoning:** Employs iterative reasoning and memory retrieval for effective task completion.
*   **M3-Bench Dataset:** Introduces a novel benchmark for evaluating long-term memory and reasoning capabilities in multimodal agents.
*   **State-of-the-Art Performance:** Outperforms existing models on M3-Bench and VideoMME-long.

## M3-Bench: Evaluating Long-Term Memory and Reasoning

M3-Bench is a comprehensive long video question-answering dataset designed to assess the capabilities of multimodal agents. It features two key subsets:

*   **M3-Bench-robot:** 100 real-world videos captured from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos covering diverse scenarios.

These videos, along with associated question-answer pairs, are designed to evaluate a multimodal agent’s ability to construct and reason over long-term memory effectively.

**Examples of M3-Bench:**
*   [Example 1](https://www.youtube.com/watch?v=7W0gRqCRMZQ)
*   [Example 2](https://www.youtube.com/watch?v=Efk3K4epEzg)
*   [Example 3](https://www.youtube.com/watch?v=6Unxpxy-Ct4)

![M3-Bench Statistics](figs/m3-bench-statistic.png)

## M3-Agent Architecture

![M3-Agent Architecture](figs/m3-agent.png)

M3-Agent's architecture is designed to mimic human-like cognitive processes.  It comprises two parallel processes:

*   **Memorization:** Processes video and audio to build episodic and semantic memory.
*   **Control:** Executes instructions by iteratively thinking and retrieving information from its multimodal memory graph.

## Experimental Results

![Experimental Results](figs/exp_result.png)

M3-Agent demonstrates superior performance on M3-Bench and VideoMME-long, showcasing its advanced long-term memory and reasoning capabilities.

## Run Locally

This section guides you through setting up and running M3-Agent locally. Before starting, ensure you have the API configurations in `configs/api_config.json`.

### 1. Memorization

Follow these steps to generate memory graphs. Note that you can download pre-processed data from Hugging Face to save time.

**Prerequisites:**

*   Set up the environment using `bash setup.sh`.
*   Install necessary libraries as specified.
*   Download the audio embedding model, *pretrained_eres2netv2.ckpt*, from [modelscope](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt) and save it in `models/`.
*   Download `speakerlab` from [Github](https://github.com/modelscope/3D-Speaker/tree/main/speakerlab)

1.  **Cut Video:**  Use the provided script to segment videos into 30-second clips.
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

2.  **Prepare Data:** Create a `data.jsonl` file with video information in the specified format.
    ```json
    {"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
    ```
3.  **Generate Intermediate Outputs:** Run the script to generate intermediate outputs.
    ```bash
    python m3_agent/memorization_intermediate_outputs.py \
       --data_file data/data.jsonl
    ```

4.  **Generate Memory Graphs:** Use the M3-Agent-Memorization model to generate memory graphs.
    ```bash
    python m3_agent/memorization_memory_graphs.py \
       --data_file data/data.jsonl
    ```
5.  **Memory Graph Visualization:** Visualize the generated memory graphs.
    ```bash
    python visualization.py \
       --mem_path data/memory_graphs/robot/bedroom_01.pkl \
       --clip_id 1
    ```

### 2. Control

1.  **Set up environment**: follow the setup instructions above.
2.  **Question Answering and Evaluation:** Utilize the M3-Agent-Control model to generate answers and evaluate with GPT-4o.

```bash
python m3_agent/control.py \
   --data_file data/annotations/robot.json
```

### 3. Other Models

Adapt the model inference to API calls and modify the prompts for use with other models.

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