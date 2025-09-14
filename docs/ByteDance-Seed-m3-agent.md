<div align="left">
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

# M3-Agent: A Multimodal Agent with Long-Term Memory for Advanced Reasoning

**M3-Agent** is a groundbreaking multimodal agent that can *see, listen, remember, and reason* like a human, enabling advanced understanding and interaction with its environment.  Learn more and contribute at the [original repository](https://github.com/ByteDance-Seed/m3-agent).

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

## Key Features

*   **Long-Term Memory:** M3-Agent builds and updates its memory with real-time visual and auditory input, creating both episodic and semantic memories.
*   **Entity-Centric, Multimodal Memory:**  Organizes information in a structured format for deeper environmental understanding.
*   **Iterative Reasoning:**  Performs multi-turn reasoning, retrieving relevant information from memory to accomplish tasks.
*   **Superior Performance:** Outperforms baseline models in the M3-Bench benchmark.
*   **M3-Bench Dataset:** Includes a new long-video question-answering dataset (M3-Bench) designed to evaluate multimodal agents.

## M3-Bench: Evaluating Long-Term Memory and Reasoning

M3-Bench is a comprehensive benchmark for evaluating the effectiveness of long-term memory and reasoning in multimodal agents. It features:

*   **M3-Bench-robot:** 100 real-world videos from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos covering diverse scenarios.
*   **Question-Answer Pairs:**  Designed to test human understanding, knowledge extraction, and cross-modal reasoning.

![M3-Bench Example](figs/m3-bench-example.png)
*Examples from M3-Bench. M3-Bench-robot features long videos from realistic robotic work scenarios, while M3-Bench-web expands the video diversity to support broader evaluation. The question-answering tasks are designed to assess a multimodal agent’s ability to construct consistent and reliable long-term memory, as well as to reason effectively over that memory.*

![M3-Bench Statistics](figs/m3-bench-statistic.png)

**Video Resources:**

*   Download M3-Bench-robot from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot).
*   Download M3-Bench-web video URLs from `data/annotations/web.json`.

**Intermediate Outputs & Memory Graphs:**

*   Download pre-processed intermediate outputs and memory graphs from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) and [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs) respectively, or generate them using the instructions below.

## Run M3-Agent Locally

Before you start, add your API configuration in `configs/api_config.json`.

### 1. Memorization

Generate memory graphs for each video.

*   **Prerequisites (only needed if not using downloaded intermediate outputs and memory graphs):**

    1.  Set up the environment:

    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```

    2.  Cut videos into 30-second segments using `ffmpeg`. The script is provided in the original README.

    3.  Prepare a JSONL file (`data/data.jsonl`) with video information.  Example:

    ```json
    {"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
    ```

    4.  **Generate Intermediate Outputs:**

    ```bash
    python m3_agent/memorization_intermediate_outputs.py --data_file data/data.jsonl
    ```
    5.  **Generate Memory Graphs:**

    ```bash
    python m3_agent/memorization_memory_graphs.py --data_file data/data.jsonl
    ```

    6.  **Visualize Memory Graphs:**

    ```bash
    python visualization.py --mem_path data/memory_graphs/robot/bedroom_01.pkl --clip_id 1
    ```

### 2. Control

1.  Set up the environment:

```bash
bash setup.sh
pip install transformers==4.51.0
pip install vllm==0.8.4
pip install numpy==1.26.4
```
2.  **Question Answering and Evaluation:**

```bash
python m3_agent/control.py --data_file data/annotations/robot.json
```

### 3. Other Models

To utilize other models for memory generation or question answering, adjust the model inference section using appropriate prompts.

**Prompts:**

*   **Memorization:**
    *   Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
    *   Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`
*   **Control:**
    *   GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training

*   Memorization:  See [hyc2026/sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker).
*   Control:  See [hyc2026/M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training).

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