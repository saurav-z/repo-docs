<div align="center">
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width="40%">
</div>

<h1 align="center">M3-Agent: A Multimodal Agent with Long-Term Memory</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

**M3-Agent empowers AI with human-like long-term memory, enabling advanced reasoning in multimodal environments.**  ([Original Repo](https://github.com/ByteDance-Seed/m3-agent))

## Key Features

*   **Multimodal Processing:**  Processes and understands visual and auditory inputs in real-time.
*   **Long-Term Memory:** Builds and updates both episodic and semantic memory, storing information in an entity-centric, multimodal format.
*   **Advanced Reasoning:**  Performs multi-turn, iterative reasoning to accomplish tasks, retrieving relevant information from its memory.
*   **M3-Bench Benchmark:**  Utilizes M3-Bench, a novel long-video question answering benchmark, to evaluate memory effectiveness and reasoning capabilities.
*   **State-of-the-Art Performance:** Outperforms leading baselines on M3-Bench and VideoMME-long.

## M3-Bench: Evaluating Long-Term Memory & Reasoning

M3-Bench is a comprehensive benchmark designed to assess the ability of multimodal agents to reason over long-term memory, composed of:

*   **M3-Bench-robot:** Contains 100 real-world videos recorded from a robot's perspective.
*   **M3-Bench-web:** Features 920 web-sourced videos covering a wide range of scenarios.

The benchmark uses question-answer pairs tailored to test key agent capabilities, like human understanding, knowledge extraction, and cross-modal reasoning.

<details>
  <summary>M3-Bench Examples</summary>
    <p>
    <br>
    <div align=center>
        <img src="figs/m3-bench-example.png" width=80%>
    </div>
    <br>
        M3-Bench-robot features long videos from realistic robotic work scenarios, while M3-Bench-web expands the video diversity to support broader evaluation. The question-answering tasks are designed to assess a multimodal agent’s ability to construct consistent and reliable long-term memory, as well as to reason effectively over that memory.
    </p>
</details>

<details>
  <summary>M3-Bench Statistics</summary>
  <p>
    <br>
    <div align=center>
        <img src="figs/m3-bench-statistic.png" width=80%>
    </div>
  <br>
  Statistical overview of M3-Bench benchmark. Each question may correspond to multiple question types.
  </p>
</details>

### Accessing M3-Bench

*   **M3-Bench-robot Videos:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot).
*   **M3-Bench-web Videos:** Download from video_url in `data/annotations/web.json`.

### Intermediate Outputs & Memory Graphs

*   Optional: Download intermediate outputs and memory graphs from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main).
*   Alternatively, generate them directly from the videos using the steps outlined below.

## M3-Agent Architecture

<details>
  <summary>M3-Agent Architecture</summary>
    <p>
    <br>
    <div align=center>
        <img src="figs/m3-agent.png" width=80%>
    </div>
    <br>
    The system consists of two parallel processes: memorization and control. During memorization, M3-Agent processes video and audio streams online to generate episodic and semantic memory. During control, it executes instructions by iteratively thinking and retrieving from long-term memory. The long-term memory is structured as a multimodal graph.
    </p>
</details>

## Experimental Results

<details>
  <summary>Performance Highlights</summary>
    <p>
    <br>
    <div align=center>
        <img src="figs/exp_result.png" width=80%>
    </div>
    <br>
    Results on M3-Bench-robot, M3-Bench-web, and VideoMME-long.
    </p>
</details>

## Run Locally

### Prerequisites

*   Ensure you have the necessary API configurations set up in `configs/api_config.json`.
*   A standard bash environment is required to run setup.sh.

### Steps

1.  **Setup Environment:**

    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```

2.  **Memorization**

    *The following steps are required only if you haven't downloaded *intermediate_outputs* and *memory_graphs* from huggingface or want to process other videos not from M3-Bench.*
    *   Cut Video
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
    *   Prepare Data

    Prepare a jsonl file with one video per line saved in `data/data.jsonl`

    ```json
    {"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
    ```
    *   Generate Intermediate Outputs

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

    *   Generate Memory Graphs

        - Download M3-Agent-Memorization from [huggingface](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)

        ```bash
        python m3_agent/memorization_memory_graphs.py \
           --data_file data/data.jsonl
        ```
    *   Memory Graph Visualization

        ```bash
        python visualization.py \
           --mem_path data/memory_graphs/robot/bedroom_01.pkl \
           --clip_id 1
        ```

3.  **Control**

    ```bash
    pip install transformers==4.51.0
    pip install vllm==0.8.4
    pip install numpy==1.26.4
    ```

4.  **Question Answering and Evaluation**

    *   Download M3-Agent-Control from [huggingface](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
    ```bash
    python m3_agent/control.py \
       --data_file data/annotations/robot.json
    ```
5.  **Other Models**
    If you want to prompt other models to generate memory or answer question, only need to change the model inference into api calling and use the corresponding prompt.

    Prompts:

    1.  Memorization
        *   Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
        *   Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`

    2.  Control
        *   GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training

*   **Memorization:** [https://github.com/hyc2026/sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
*   **Control:** [https://github.com/hyc2026/M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

## Citation

If you use this work, please cite us as:

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