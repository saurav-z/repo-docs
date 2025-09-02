<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

# M3-Agent: Building Human-Like Long-Term Memory for Multimodal Agents

**M3-Agent** is a cutting-edge multimodal agent capable of understanding and interacting with the world through sight, sound, and long-term memory, enabling advanced reasoning capabilities. Find the original repo [here](https://github.com/ByteDance-Seed/m3-agent).

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

## Key Features

*   **Multimodal Understanding:** Processes visual and auditory inputs in real-time.
*   **Long-Term Memory:** Builds and updates an entity-centric, multimodal memory for comprehensive world knowledge.
*   **Advanced Reasoning:**  Performs iterative reasoning and retrieves relevant information from memory to complete tasks.
*   **State-of-the-Art Performance:** Outperforms leading baselines on the M3-Bench benchmark.
*   **M3-Bench Benchmark:**  A new benchmark designed to evaluate memory effectiveness and reasoning in multimodal agents.

## What is M3-Agent?

M3-Agent is designed to mimic human-like cognitive abilities by integrating perception, memory, and reasoning. The system is comprised of two processes: memorization and control, allowing the agent to build a comprehensive understanding of its environment and reason through multi-turn interactions.

### How it Works

1.  **Memorization:** The agent processes real-time video and audio streams to generate episodic and semantic memory.
2.  **Control:** Given an instruction, the agent iteratively thinks, retrieves information from its long-term memory, and performs actions.

### [Demo](https://www.youtube.com/watch?v=XUx31cBanfo)

Watch M3-Agent in action as a personal assistant!
The video is also accessible on [Bilibili](https://www.bilibili.com/video/BV1h9YpznEx9/)

### M3-Bench: Evaluating Long-Term Memory and Reasoning

M3-Bench is a novel long-video question-answering benchmark designed to evaluate the capabilities of multimodal agents. It includes two subsets:

*   **M3-Bench-robot:** 100 real-world videos recorded from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos across diverse scenarios.

**M3-Bench provides a platform to assess the agent's ability to build consistent long-term memory and reason effectively over that memory.**

![architecture](figs/m3-bench-example.png)

[link1](https://www.youtube.com/watch?v=7W0gRqCRMZQ), [link2](https://www.youtube.com/watch?v=Efk3K4epEzg), [link3](https://www.youtube.com/watch?v=6Unxpxy-Ct4)
Examples from M3-Bench. M3-Bench-robot features long videos from realistic robotic work scenarios, while M3-Bench-web expands the video diversity to support broader evaluation. The question-answering tasks are designed to assess a multimodal agent’s ability to construct consistent and reliable long-term memory, as well as to reason effectively over that memory.

![architecture](figs/m3-bench-statistic.png)

Statistical overview of M3-Bench benchmark. Each question may correspond to multiple question types.

### Accessing the Data

*   **M3-Bench-robot videos:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)
*   **M3-Bench-web videos:** Accessible via the `video_url` in `data/annotations/web.json`.
*   **Intermediate Outputs:**  Optionally download processed intermediate outputs from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs).
*   **Memory Graphs:** Download pre-processed memory graphs from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs), or generate them from the videos (see instructions below).

## Run Locally

### Prerequisites
*   Add API configuration in `configs/api_config.json`
*   Ensure you have the necessary models and libraries downloaded and installed.

### Memorization

These steps are required only if you haven't downloaded *intermediate_outputs* and *memory_graphs* from huggingface or want to process other videos not from M3-Bench.

1.  **Set up the Environment:**
    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```

2.  **Cut Video**

    Cut the video into 30 second segments.
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

    Create a `data/data.jsonl` file with the following format:

    ```json
    {"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
    ```

4.  **Generate Intermediate Outputs:**

    **Note:** This step uses face detection and speaker diarization tools to generate intermediate outputs. If you've downloaded the intermediate outputs from Hugging Face, you can skip this step.

    *   Download the audio embedding model `pretrained_eres2netv2.ckpt` into `models\` from [pretrained_eres2netv2.ckpt](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt)
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

    **Note:** This step uses the M3-Agent-Memorization model.  Download the model from [Hugging Face](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization).

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

1.  **Set up the Environment:**
    ```bash
    bash setup.sh
    pip install transformers==4.51.0
    pip install vllm==0.8.4
    pip install numpy==1.26.4
    ```

2.  **Question Answering and Evaluation:**

    **Note:** This step uses the M3-Agent-Control model to generate answers and GPT-4o for evaluation. Download the M3-Agent-Control model from [Hugging Face](https://huggingface.co/ByteDance-Seed/M3-Agent-Control).

    ```bash
    python m3_agent/control.py \
        --data_file data/annotations/robot.json
    ```

### Other Models

You can adapt the code to use different models by modifying the model inference calls with the corresponding prompts.

**Prompts:**

1.  **Memorization:**
    *   Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
    *   Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`

2.  **Control:**
    *   GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training

*   Memorization: Refer to [sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
*   Control: Refer to [M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

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