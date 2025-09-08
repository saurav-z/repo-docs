<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: A Multimodal Agent with Human-Like Long-Term Memory</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

**M3-Agent is a groundbreaking multimodal agent designed to perceive, remember, and reason, mirroring human-like cognitive abilities.**

[View the original repository on GitHub](https://github.com/ByteDance-Seed/m3-agent)

## Key Features

*   **Multimodal Perception:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Builds and updates both episodic and semantic memory.
*   **Entity-Centric Memory:** Organizes memory in a multimodal graph format for a deeper understanding of the environment.
*   **Iterative Reasoning:**  Performs multi-turn reasoning and retrieves relevant information to accomplish tasks.
*   **M3-Bench Dataset:**  Provides a novel long-video question-answering benchmark for evaluating memory effectiveness.
*   **Superior Performance:** Outperforms leading baselines on the M3-Bench dataset.

## What is M3-Agent?

M3-Agent represents a significant advancement in multimodal agents, offering a framework capable of human-like long-term memory and reasoning. It excels in understanding and interacting with complex environments, setting a new standard for agent capabilities.

### M3-Bench: Evaluating Long-Term Memory

M3-Bench is a novel benchmark designed to assess the memory and reasoning capabilities of multimodal agents. It includes two subsets:

*   **M3-Bench-robot:** Features 100 real-world videos from a robot's perspective.
*   **M3-Bench-web:** Contains 920 web-sourced videos across diverse scenarios.

This benchmark challenges agents to answer questions based on long-term memory, providing a comprehensive evaluation of their abilities.

![M3-Bench Example](figs/m3-bench-example.png)
*(Example from the M3-Bench dataset. Demonstrates the kind of question-answering tasks performed.)*

![M3-Bench Statistics](figs/m3-bench-statistic.png)
*(Statistical overview of the M3-Bench benchmark.)*

### Accessing the Data

*   **M3-Bench-robot:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)
*   **M3-Bench-web:** Download from video_url in `data/annotations/web.json`

## M3-Agent Architecture

![M3-Agent Architecture](figs/m3-agent.png)

M3-Agent's architecture comprises two parallel processes:

*   **Memorization:** Processes video and audio streams to generate episodic and semantic memory.
*   **Control:** Executes instructions through iterative thinking and memory retrieval, structured as a multimodal graph.

## Experimental Results

![Experimental Results](figs/exp_result.png)

M3-Agent showcases impressive performance on M3-Bench-robot, M3-Bench-web, and VideoMME-long, demonstrating its effectiveness in complex reasoning tasks.

## Run Locally

Follow these instructions to run M3-Agent locally:

**Prerequisites:**

*   Add API configuration in `configs/api_config.json`
*   Set up the environment with `bash setup.sh` and install the required packages.

### 1. Memorization
Generate memory graphs for each video. The results are saved in `data/memory_graphs`.

**Steps (If not using pre-processed data):**

1.  **Set up environment:**
    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```

2.  **Cut Video:** Divide videos into 30-second segments using the provided `cut_video.sh` script.  (See original README for script.)

3.  **Prepare Data:** Create a JSONL file (`data/data.jsonl`) with video metadata (See original README for example.)

4.  **Generate Intermediate Outputs:** (Requires Face Detection and Speaker Diarization tools.  Skip this step if you've downloaded `intermediate_outputs` from Hugging Face.)
    *   Download the audio embedding model and save it into `models/` (link provided in original README)
    *   Download speakerlab and save into  `speakerlab/` (link provided in original README)

    ```bash
    python m3_agent/memorization_intermediate_outputs.py \
       --data_file data/data.jsonl
    ```

5.  **Generate Memory Graphs:** (Requires the M3-Agent-Memorization model from Hugging Face.)

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

1.  **Set up environment:**

    ```bash
    bash setup.sh
    pip install transformers==4.51.0
    pip install vllm==0.8.4
    pip install numpy==1.26.4
    ```

2.  **Question Answering and Evaluation:** (Requires the M3-Agent-Control model from Hugging Face.)

    ```bash
    python m3_agent/control.py \
       --data_file data/annotations/robot.json
    ```

### 3. Other Models

Adapt the model inference by calling APIs using different prompts:

*   **Memorization Prompts:**
    *   Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
    *   Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`

*   **Control Prompts:**
    *   GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training

Training code is available at the following repositories:

1.  Memorization: [sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
2.  Control: [M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

## Citation

If you use this work, please cite:

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