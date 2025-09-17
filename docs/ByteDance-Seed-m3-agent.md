<div align="left">
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

# M3-Agent: The Multimodal Agent with Long-Term Memory

**M3-Agent** is a groundbreaking multimodal agent that sees, listens, remembers, and reasons, bringing us closer to human-like AI.  Find out more on the original repository: [https://github.com/ByteDance-Seed/m3-agent](https://github.com/ByteDance-Seed/m3-agent).

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

## Key Features of M3-Agent:

*   **Long-Term Memory:**  Processes visual and auditory input to build and update a robust long-term memory, including episodic and semantic components.
*   **Entity-Centric Memory:**  Organizes memory in a multimodal, entity-centric format, enabling deeper understanding of the environment.
*   **Iterative Reasoning:**  Performs multi-turn, iterative reasoning and retrieves relevant information from memory to accomplish tasks.
*   **Superior Performance:** Outperforms leading baselines, achieving higher accuracy on the M3-Bench benchmark.
*   **Versatile Application:** Provides insights into practical design and advances multimodal agents towards more human-like memory capabilities.

## M3-Bench: Evaluating Multimodal Reasoning

M3-Agent's capabilities are evaluated using the **M3-Bench**, a dedicated benchmark designed to assess memory effectiveness and reasoning abilities in multimodal agents.  It consists of:

*   **M3-Bench-robot:** 100 real-world videos recorded from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos across diverse scenarios.

The benchmark includes question-answer pairs testing key agent capabilities like human understanding, general knowledge extraction, and cross-modal reasoning.

![architecture](figs/m3-bench-example.png)
*Examples from M3-Bench.  M3-Bench-robot features long videos from realistic robotic work scenarios, while M3-Bench-web expands the video diversity to support broader evaluation. The question-answering tasks are designed to assess a multimodal agent’s ability to construct consistent and reliable long-term memory, as well as to reason effectively over that memory.*

![architecture](figs/m3-bench-statistic.png)
*Statistical overview of M3-Bench benchmark. Each question may correspond to multiple question types.*

### Data Downloads

1.  **M3-Bench-robot Videos:** [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)
2.  **M3-Bench-web Videos:** Download from `video_url` in `data/annotations/web.json`

### Intermediate Outputs & Memory Graphs

**[Optional]** Download pre-processed intermediate outputs and memory graphs from Hugging Face for faster setup:

*   **Intermediate Outputs:** [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs)
*   **Memory Graphs:** [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs)

Alternatively, you can generate them from the video using the steps outlined below.

## M3-Agent Architecture

![architecture](figs/m3-agent.png)

*Architecture of M3-Agent. The system consists of two parallel processes: memorization and control. During memorization, M3-Agent processes video and audio streams online to generate episodic and semantic memory. During control, it executes instructions by iteratively thinking and retrieving from long-term memory. The long-term memory is structured as a multimodal graph.*

M3-Agent employs a unique architecture with two key processes:

1.  **Memorization:** Processes video and audio data to create episodic and semantic memory.
2.  **Control:** Executes instructions by iteratively reasoning and retrieving information from its long-term, multimodal graph-structured memory.

## Experimental Results

![architecture](figs/exp_result.png)
*Results on M3-Bench-robot, M3-Bench-web, and VideoMME-long.*

## Running M3-Agent Locally

These instructions will get you up and running with M3-Agent.

**Important:** Before running, add your API configuration to `configs/api_config.json`.

### Memorization

1.  **Set up Environment:**

    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```

2.  **Cut Video (if not using pre-cut clips):**

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

3.  **Prepare Data:** Create a `data/data.jsonl` file with one line per video in JSONL format:

    ```json
    {"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
    ```

4.  **Generate Intermediate Outputs:**

    **Requires Face Detection and Speaker Diarization tools.**
    * If you are using pre-generated outputs from Hugging Face, you can skip this step.
    * Download audio embedding model into `models\` from [pretrained_eres2netv2.ckpt](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt)
    * Download [speakerlab](https://github.com/modelscope/3D-Speaker/tree/main/speakerlab)

   ```bash
   python m3_agent/memorization_intermediate_outputs.py \
      --data_file data/data.jsonl
   ```

5.  **Generate Memory Graphs:**

    **Requires the M3-Agent-Memorization model.** Download from [Hugging Face](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization).

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

1.  **Set up Environment:**

    ```bash
    bash setup.sh
    pip install transformers==4.51.0
    pip install vllm==0.8.4
    pip install numpy==1.26.4
    ```

2.  **Question Answering and Evaluation:**

    **Requires the M3-Agent-Control model to generate answers and GPT-4o to evaluate the answer.** Download M3-Agent-Control from [Hugging Face](https://huggingface.co/ByteDance-Seed/M3-Agent-Control).

    ```bash
    python m3_agent/control.py \
       --data_file data/annotations/robot.json
    ```

### Prompting Other Models

Adapt the model inference by calling the API and using the corresponding prompts:

1.  **Memorization Prompts:**
    *   Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
    *   Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`

2.  **Control Prompts:**
    *   GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training

*   **Memorization:** [https://github.com/hyc2026/sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
*   **Control:** [https://github.com/hyc2026/M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

## Citation

If you use M3-Agent, please cite the following:

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